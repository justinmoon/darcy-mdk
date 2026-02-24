//! Snapshot and rollback support for memory storage.
//!
//! This module provides the ability to create snapshots of all in-memory state
//! and restore them later. This provides functionality analogous to SQLite
//! savepoints for testing and rollback scenarios.
//!
//! # Concurrency
//!
//! Snapshot creation and restoration are **atomic** operations:
//!
//! - `create_snapshot()` acquires a global read lock on the storage state,
//!   ensuring a consistent snapshot even with concurrent reads.
//! - `restore_snapshot()` acquires a global write lock on the storage state,
//!   ensuring the restore is consistent and blocks all other operations.

use std::collections::{BTreeSet, HashMap};

use lru::LruCache;
use mdk_storage_traits::GroupId;
use mdk_storage_traits::groups::types::{Group, GroupExporterSecret, GroupRelay};
use mdk_storage_traits::messages::types::{Message, ProcessedMessage};
use mdk_storage_traits::welcomes::types::{ProcessedWelcome, Welcome};
use mdk_storage_traits::MdkStorageError;
use nostr::EventId;
use serde::{Deserialize, Serialize};

use crate::mls_storage::GroupDataType;

/// A group-scoped snapshot that only contains data for a single group.
///
/// Unlike [`MemoryStorageSnapshot`] which captures all data in the storage,
/// this snapshot only captures data relevant to a specific group. This enables
/// proper rollback isolation where rolling back Group A doesn't affect Group B.
///
/// This matches the behavior of SQLite's group-scoped snapshots where:
/// - `snapshot_group_state()` only copies rows WHERE `group_id = ?`
/// - `restore_group_from_snapshot()` only deletes/restores rows for that group
///
/// # Group-Scoped Data
///
/// The following data is captured per group:
/// - MLS group data (tree state, join config, etc.)
/// - MLS own leaf nodes for this group
/// - MLS proposals for this group
/// - MLS epoch key pairs for this group
/// - MDK group record
/// - MDK group relays
/// - MDK group exporter secrets
///
/// The following data is NOT captured (not group-scoped):
/// - MLS key packages (identity-scoped, not group-scoped)
/// - MLS PSKs (identity-scoped, not group-scoped)
/// - MLS signature keys (identity-scoped, not group-scoped)
/// - MLS encryption keys (identity-scoped, not group-scoped)
/// - Messages (handled separately via `invalidate_messages_after_epoch`)
/// - Welcomes (keyed by EventId, not group-scoped)
#[derive(Clone)]
pub struct GroupScopedSnapshot {
    /// The group ID this snapshot is for
    pub(crate) group_id: GroupId,

    /// Unix timestamp when this snapshot was created
    pub(crate) created_at: u64,

    // MLS data (filtered by group_id)
    /// MLS group data: (group_id, data_type) -> data
    pub(crate) mls_group_data: HashMap<(Vec<u8>, GroupDataType), Vec<u8>>,
    /// MLS own leaf nodes for this group
    pub(crate) mls_own_leaf_nodes: Vec<Vec<u8>>,
    /// MLS proposals: proposal_ref -> proposal (group_id is implicit)
    pub(crate) mls_proposals: HashMap<Vec<u8>, Vec<u8>>,
    /// MLS epoch key pairs: (epoch_id, leaf_index) -> key_pairs (group_id is implicit)
    pub(crate) mls_epoch_key_pairs: HashMap<(Vec<u8>, u32), Vec<u8>>,

    // MDK data
    /// The group record itself
    pub(crate) group: Option<Group>,
    /// Group relays
    pub(crate) group_relays: BTreeSet<GroupRelay>,
    /// Group exporter secrets: epoch -> secret
    pub(crate) group_exporter_secrets: HashMap<u64, GroupExporterSecret>,
}

/// A snapshot of all in-memory state that can be restored later.
///
/// This enables rollback functionality similar to SQLite savepoints,
/// allowing you to:
/// 1. Create a snapshot before an operation
/// 2. Attempt the operation
/// 3. Restore the snapshot if the operation fails or needs to be undone
///
/// # Concurrency
///
/// Snapshot creation and restoration are **atomic**. `create_snapshot()` acquires
/// a global read lock and `restore_snapshot()` acquires a global write lock,
/// ensuring consistency in multi-threaded environments.
///
/// # Example
///
/// ```ignore
/// let storage = MdkMemoryStorage::default();
///
/// // Make some changes
/// storage.save_group(group)?;
///
/// // Create a snapshot (ensure no concurrent operations)
/// let snapshot = storage.create_snapshot();
///
/// // Try an operation that might need rollback
/// storage.save_message(message)?;
///
/// // If we need to undo (ensure no concurrent operations):
/// storage.restore_snapshot(snapshot);
/// ```
#[derive(Clone)]
pub struct MemoryStorageSnapshot {
    // MLS data
    pub(crate) mls_group_data: HashMap<(Vec<u8>, GroupDataType), Vec<u8>>,
    pub(crate) mls_own_leaf_nodes: HashMap<Vec<u8>, Vec<Vec<u8>>>,
    pub(crate) mls_proposals: HashMap<(Vec<u8>, Vec<u8>), Vec<u8>>,
    pub(crate) mls_key_packages: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) mls_psks: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) mls_signature_keys: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) mls_encryption_keys: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) mls_epoch_key_pairs: HashMap<(Vec<u8>, Vec<u8>, u32), Vec<u8>>,

    // MDK data - cloned from LRU caches
    pub(crate) groups: HashMap<GroupId, Group>,
    pub(crate) groups_by_nostr_id: HashMap<[u8; 32], Group>,
    pub(crate) group_relays: HashMap<GroupId, BTreeSet<GroupRelay>>,
    pub(crate) group_exporter_secrets: HashMap<(GroupId, u64), GroupExporterSecret>,
    pub(crate) welcomes: HashMap<EventId, Welcome>,
    pub(crate) processed_welcomes: HashMap<EventId, ProcessedWelcome>,
    pub(crate) messages: HashMap<EventId, Message>,
    pub(crate) messages_by_group: HashMap<GroupId, HashMap<EventId, Message>>,
    pub(crate) processed_messages: HashMap<EventId, ProcessedMessage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct SerializableMemoryStorageSnapshot {
    // MLS data
    pub mls_group_data: Vec<(Vec<u8>, u8, Vec<u8>)>,
    pub mls_own_leaf_nodes: HashMap<Vec<u8>, Vec<Vec<u8>>>,
    pub mls_proposals: HashMap<(Vec<u8>, Vec<u8>), Vec<u8>>,
    pub mls_key_packages: HashMap<Vec<u8>, Vec<u8>>,
    pub mls_psks: HashMap<Vec<u8>, Vec<u8>>,
    pub mls_signature_keys: HashMap<Vec<u8>, Vec<u8>>,
    pub mls_encryption_keys: HashMap<Vec<u8>, Vec<u8>>,
    pub mls_epoch_key_pairs: HashMap<(Vec<u8>, Vec<u8>, u32), Vec<u8>>,

    // MDK data
    pub groups: HashMap<GroupId, Group>,
    pub groups_by_nostr_id: HashMap<[u8; 32], Group>,
    pub group_relays: HashMap<GroupId, BTreeSet<GroupRelay>>,
    pub group_exporter_secrets: HashMap<(GroupId, u64), GroupExporterSecret>,
    pub welcomes: HashMap<EventId, Welcome>,
    pub processed_welcomes: HashMap<EventId, ProcessedWelcome>,
    pub messages: HashMap<EventId, Message>,
    pub messages_by_group: HashMap<GroupId, HashMap<EventId, Message>>,
    pub processed_messages: HashMap<EventId, ProcessedMessage>,
}

impl TryFrom<MemoryStorageSnapshot> for SerializableMemoryStorageSnapshot {
    type Error = MdkStorageError;

    fn try_from(snapshot: MemoryStorageSnapshot) -> Result<Self, Self::Error> {
        let mut mls_group_data = Vec::with_capacity(snapshot.mls_group_data.len());
        for ((group_id, data_type), data) in snapshot.mls_group_data {
            mls_group_data.push((group_id, encode_group_data_type(data_type), data));
        }
        Ok(Self {
            mls_group_data,
            mls_own_leaf_nodes: snapshot.mls_own_leaf_nodes,
            mls_proposals: snapshot.mls_proposals,
            mls_key_packages: snapshot.mls_key_packages,
            mls_psks: snapshot.mls_psks,
            mls_signature_keys: snapshot.mls_signature_keys,
            mls_encryption_keys: snapshot.mls_encryption_keys,
            mls_epoch_key_pairs: snapshot.mls_epoch_key_pairs,
            groups: snapshot.groups,
            groups_by_nostr_id: snapshot.groups_by_nostr_id,
            group_relays: snapshot.group_relays,
            group_exporter_secrets: snapshot.group_exporter_secrets,
            welcomes: snapshot.welcomes,
            processed_welcomes: snapshot.processed_welcomes,
            messages: snapshot.messages,
            messages_by_group: snapshot.messages_by_group,
            processed_messages: snapshot.processed_messages,
        })
    }
}

impl TryFrom<SerializableMemoryStorageSnapshot> for MemoryStorageSnapshot {
    type Error = MdkStorageError;

    fn try_from(snapshot: SerializableMemoryStorageSnapshot) -> Result<Self, Self::Error> {
        let mut mls_group_data = HashMap::with_capacity(snapshot.mls_group_data.len());
        for (group_id, data_type, data) in snapshot.mls_group_data {
            mls_group_data.insert((group_id, decode_group_data_type(data_type)?), data);
        }
        Ok(Self {
            mls_group_data,
            mls_own_leaf_nodes: snapshot.mls_own_leaf_nodes,
            mls_proposals: snapshot.mls_proposals,
            mls_key_packages: snapshot.mls_key_packages,
            mls_psks: snapshot.mls_psks,
            mls_signature_keys: snapshot.mls_signature_keys,
            mls_encryption_keys: snapshot.mls_encryption_keys,
            mls_epoch_key_pairs: snapshot.mls_epoch_key_pairs,
            groups: snapshot.groups,
            groups_by_nostr_id: snapshot.groups_by_nostr_id,
            group_relays: snapshot.group_relays,
            group_exporter_secrets: snapshot.group_exporter_secrets,
            welcomes: snapshot.welcomes,
            processed_welcomes: snapshot.processed_welcomes,
            messages: snapshot.messages,
            messages_by_group: snapshot.messages_by_group,
            processed_messages: snapshot.processed_messages,
        })
    }
}

pub(crate) fn snapshot_to_bytes(snapshot: MemoryStorageSnapshot) -> Result<Vec<u8>, MdkStorageError> {
    let serializable = SerializableMemoryStorageSnapshot::try_from(snapshot)?;
    bincode::serialize(&serializable)
        .map_err(|e| MdkStorageError::Serialization(format!("serialize snapshot: {e}")))
}

pub(crate) fn snapshot_from_bytes(bytes: &[u8]) -> Result<MemoryStorageSnapshot, MdkStorageError> {
    let serializable: SerializableMemoryStorageSnapshot = bincode::deserialize(bytes)
        .map_err(|e| MdkStorageError::Deserialization(format!("deserialize snapshot: {e}")))?;
    MemoryStorageSnapshot::try_from(serializable)
}

fn encode_group_data_type(data_type: GroupDataType) -> u8 {
    match data_type {
        GroupDataType::JoinGroupConfig => 1,
        GroupDataType::Tree => 2,
        GroupDataType::InterimTranscriptHash => 3,
        GroupDataType::Context => 4,
        GroupDataType::ConfirmationTag => 5,
        GroupDataType::GroupState => 6,
        GroupDataType::MessageSecrets => 7,
        GroupDataType::ResumptionPskStore => 8,
        GroupDataType::OwnLeafIndex => 9,
        GroupDataType::GroupEpochSecrets => 10,
    }
}

fn decode_group_data_type(code: u8) -> Result<GroupDataType, MdkStorageError> {
    match code {
        1 => Ok(GroupDataType::JoinGroupConfig),
        2 => Ok(GroupDataType::Tree),
        3 => Ok(GroupDataType::InterimTranscriptHash),
        4 => Ok(GroupDataType::Context),
        5 => Ok(GroupDataType::ConfirmationTag),
        6 => Ok(GroupDataType::GroupState),
        7 => Ok(GroupDataType::MessageSecrets),
        8 => Ok(GroupDataType::ResumptionPskStore),
        9 => Ok(GroupDataType::OwnLeafIndex),
        10 => Ok(GroupDataType::GroupEpochSecrets),
        _ => Err(MdkStorageError::Deserialization(format!(
            "unknown GroupDataType code {code}"
        ))),
    }
}

#[cfg(test)]
impl MemoryStorageSnapshot {
    /// Create a new empty snapshot for testing.
    pub(crate) fn new() -> Self {
        Self {
            mls_group_data: HashMap::new(),
            mls_own_leaf_nodes: HashMap::new(),
            mls_proposals: HashMap::new(),
            mls_key_packages: HashMap::new(),
            mls_psks: HashMap::new(),
            mls_signature_keys: HashMap::new(),
            mls_encryption_keys: HashMap::new(),
            mls_epoch_key_pairs: HashMap::new(),
            groups: HashMap::new(),
            groups_by_nostr_id: HashMap::new(),
            group_relays: HashMap::new(),
            group_exporter_secrets: HashMap::new(),
            welcomes: HashMap::new(),
            processed_welcomes: HashMap::new(),
            messages: HashMap::new(),
            messages_by_group: HashMap::new(),
            processed_messages: HashMap::new(),
        }
    }
}

/// Helper trait to clone LRU cache contents into a HashMap.
pub(crate) trait LruCacheExt<K, V> {
    /// Clone all entries from the LRU cache into a HashMap.
    fn clone_to_hashmap(&self) -> HashMap<K, V>
    where
        K: Clone + std::hash::Hash + Eq,
        V: Clone;
}

impl<K, V> LruCacheExt<K, V> for LruCache<K, V> {
    fn clone_to_hashmap(&self) -> HashMap<K, V>
    where
        K: Clone + std::hash::Hash + Eq,
        V: Clone,
    {
        self.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

/// Helper trait to restore HashMap contents back into an LRU cache.
pub(crate) trait HashMapToLruExt<K, V> {
    /// Restore entries from a HashMap into an LRU cache.
    fn restore_to_lru(&self, cache: &mut LruCache<K, V>)
    where
        K: Clone + std::hash::Hash + Eq,
        V: Clone;
}

impl<K, V> HashMapToLruExt<K, V> for HashMap<K, V> {
    fn restore_to_lru(&self, cache: &mut LruCache<K, V>)
    where
        K: Clone + std::hash::Hash + Eq,
        V: Clone,
    {
        cache.clear();
        for (k, v) in self.iter() {
            cache.put(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    #[test]
    fn test_lru_cache_clone_to_hashmap() {
        let mut cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.put("c".to_string(), 3);

        let map = cache.clone_to_hashmap();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("b"), Some(&2));
        assert_eq!(map.get("c"), Some(&3));
    }

    #[test]
    fn test_hashmap_restore_to_lru() {
        let mut map = HashMap::new();
        map.insert("x".to_string(), 10);
        map.insert("y".to_string(), 20);

        let mut cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put("old".to_string(), 999);

        map.restore_to_lru(&mut cache);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"x".to_string()), Some(&10));
        assert_eq!(cache.get(&"y".to_string()), Some(&20));
        assert!(cache.get(&"old".to_string()).is_none());
    }

    #[test]
    fn test_empty_snapshot() {
        let snapshot = MemoryStorageSnapshot::new();
        assert!(snapshot.mls_group_data.is_empty());
        assert!(snapshot.groups.is_empty());
        assert!(snapshot.messages.is_empty());
    }
}
