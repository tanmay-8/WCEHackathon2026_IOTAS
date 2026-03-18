"""
Update Handler Service - Handle corrections and updates to existing graph nodes.

Strategies:
1. UPDATE: Modify existing node properties (replace old with new)
2. DEPRECATE: Mark old node as outdated, create new superceding version
3. MERGE: Combine multiple nodes into one (when duplicates found)
4. VERSION: Keep both versions, link with SUPERSEDES relationship (audit trail)
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from neo4j import GraphDatabase
from config.settings import Settings


class UpdateHandler:
    """
    Handles updating, merging, and versioning of graph entities.
    
    Update Semantics:
    - UPDATE: Direct property modification (fast, lossy)
    - DEPRECATE: Mark old as outdated, keep for audit (safe)
    - VERSION: Create version chain (CURRENT -> PREVIOUS -> ...)
    - MERGE: Combine duplicates (smart deduplication)
    """
    
    # Version control settings
    VERSION_HISTORY_DEPTH = 5  # Keep last 5 versions
    AUDIT_ENABLED = True  # Track all changes
    
    def __init__(self):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                Settings.NEO4J_URI,
                auth=(Settings.NEO4J_USER, Settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            self.driver = None
    
    def update_node(
        self,
        user_id: str,
        node_id: str,
        node_type: str,
        new_properties: Dict[str, Any],
        update_strategy: str = "update",
        justification: str = "",
        preserve_history: bool = True
    ) -> Dict[str, Any]:
        """
        Update an existing node with new properties.
        
        Args:
            user_id: User identifier
            node_id: ID of node to update
            node_type: Type of node (Asset, Transaction, etc.)
            new_properties: New property values
            update_strategy: "update"|"deprecate"|"version"
            justification: Why this update (from correction)
            preserve_history: Whether to keep audit trail
            
        Returns:
            {
                "success": bool,
                "node_id": str,
                "version": int,  # New version number
                "updated_properties": {...},
                "previous_values": {...},
                "strategy_used": str,
                "message": str
            }
        """
        if not self.driver:
            return {
                "success": False,
                "message": "Database not initialized"
            }
        
        if update_strategy == "deprecate":
            return self._deprecate_and_create(
                user_id, node_id, node_type, new_properties, justification
            )
        elif update_strategy == "version":
            return self._version_update(
                user_id, node_id, node_type, new_properties, justification
            )
        else:  # "update" strategy (default)
            return self._direct_update(
                user_id, node_id, node_type, new_properties, justification, preserve_history
            )
    
    def _direct_update(
        self,
        user_id: str,
        node_id: str,
        node_type: str,
        new_properties: Dict[str, Any],
        justification: str,
        preserve_history: bool
    ) -> Dict[str, Any]:
        """
        Direct update: Modify node properties in place.
        
        Fastest but loses history unless audit trail is enabled.
        """
        if not self.driver:
            return {"success": False}
        
        try:
            with self.driver.session() as session:
                # 1. Get current properties (for history)
                get_query = """
                MATCH (n {id: $node_id, user_id: $user_id, type: $node_type})
                RETURN properties(n) as current_props, n.version as version
                """
                
                result = session.run(
                    get_query,
                    node_id=node_id,
                    user_id=user_id,
                    node_type=node_type
                )
                
                record = result.single()
                if not record:
                    return {
                        "success": False,
                        "message": f"Node {node_id} not found"
                    }
                
                current_props = dict(record.get("current_props", {}))
                current_version = record.get("version", 0)
                new_version = current_version + 1
                
                # 2. Preserve system fields
                new_properties["last_updated"] = datetime.utcnow().isoformat()
                new_properties["updated_by"] = "user_correction"
                new_properties["version"] = new_version
                
                # 3. Create audit entry if enabled
                if preserve_history and self.AUDIT_ENABLED:
                    new_properties["update_history"] = [
                        {
                            "version": current_version,
                            "timestamp": datetime.utcnow().isoformat(),
                            "values": current_props,
                            "justification": justification if justification else "Automatic update"
                        }
                    ]
                
                # 4. Update node
                set_clause = ", ".join([f"n.{k} = ${k}" for k in new_properties.keys()])
                update_query = f"""
                MATCH (n {{id: $node_id, user_id: $user_id, type: $node_type}})
                SET {set_clause}
                RETURN n.id as node_id, n.version as version
                """
                
                params = {
                    "node_id": node_id,
                    "user_id": user_id,
                    "node_type": node_type,
                    **new_properties
                }
                
                result = session.run(update_query, **params)
                
                if result.consume().counters.properties_set > 0:
                    return {
                        "success": True,
                        "node_id": node_id,
                        "version": new_version,
                        "updated_properties": new_properties,
                        "previous_values": current_props,
                        "strategy_used": "direct_update",
                        "message": f"Updated {node_type} node {node_id} to version {new_version}"
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to update node"
                    }
        
        except Exception as e:
            print(f"[UpdateHandler] Direct update error: {e}")
            return {
                "success": False,
                "message": f"Update error: {str(e)}"
            }
    
    def _version_update(
        self,
        user_id: str,
        node_id: str,
        node_type: str,
        new_properties: Dict[str, Any],
        justification: str
    ) -> Dict[str, Any]:
        """
        Version update: Create version chain with SUPERSEDES relationship.
        
        Keeps full history: new version -> SUPERSEDES -> old version -> ...
        """
        if not self.driver:
            return {"success": False}
        
        try:
            with self.driver.session() as session:
                # 1. Get current node
                get_query = """
                MATCH (current {id: $node_id, user_id: $user_id, type: $node_type})
                RETURN properties(current) as current_props, current.version as version
                """
                
                result = session.run(
                    get_query,
                    node_id=node_id,
                    user_id=user_id,
                    node_type=node_type
                )
                
                record = result.single()
                if not record:
                    return {"success": False, "message": "Node not found"}
                
                current_version = record.get("version", 0)
                
                # 2. Create new version node (copy of current with updates)
                new_id = str(uuid.uuid4())
                new_version_num = current_version + 1
                
                # Copy properties and add version fields
                versioned_props = dict(record.get("current_props", {}))
                versioned_props.update(new_properties)
                versioned_props["id"] = new_id
                versioned_props["version"] = new_version_num
                versioned_props["created_at"] = datetime.utcnow().isoformat()
                versioned_props["supersedes_id"] = node_id
                versioned_props["update_justification"] = justification
                
                # 3. Create node creation query
                prop_str = ", ".join([f"{k}: ${k}" for k in versioned_props.keys()])
                create_query = f"""
                CREATE (v:{node_type} {{{prop_str}}})
                WITH v
                MATCH (previous {{id: $previous_id, type: $node_type, user_id: $user_id}})
                CREATE (v)-[:SUPERSEDES]->(previous)
                RETURN v.id as new_id, v.version as new_version
                """
                
                params = {
                    "previous_id": node_id,
                    "node_type": node_type,
                    "user_id": user_id,
                    **versioned_props
                }
                
                result = session.run(create_query, **params)
                version_result = result.single()
                
                if version_result:
                    return {
                        "success": True,
                        "node_id": version_result.get("new_id"),
                        "version": version_result.get("new_version"),
                        "updated_properties": new_properties,
                        "strategy_used": "version_update",
                        "message": f"Created new version {version_result.get('new_version')} (supersedes {current_version})"
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to create version"
                    }
        
        except Exception as e:
            print(f"[UpdateHandler] Version update error: {e}")
            return {
                "success": False,
                "message": f"Version error: {str(e)}"
            }
    
    def _deprecate_and_create(
        self,
        user_id: str,
        old_node_id: str,
        node_type: str,
        new_properties: Dict[str, Any],
        justification: str
    ) -> Dict[str, Any]:
        """
        Deprecate old node and create new one.
        
        Used when asset is sold, goal changed, etc.
        """
        if not self.driver:
            return {"success": False}
        
        try:
            with self.driver.session() as session:
                # 1. Mark old node as deprecated
                deprecate_query = """
                MATCH (old {id: $old_node_id, user_id: $user_id, type: $node_type})
                SET old.deprecated = true,
                    old.deprecated_at = $timestamp,
                    old.deprecation_reason = $reason
                RETURN old.id as old_id
                """
                
                timestamp = datetime.utcnow().isoformat()
                session.run(
                    deprecate_query,
                    old_node_id=old_node_id,
                    user_id=user_id,
                    node_type=node_type,
                    timestamp=timestamp,
                    reason=justification
                )
                
                # 2. Create new node
                new_id = str(uuid.uuid4())
                new_properties["id"] = new_id
                new_properties["user_id"] = user_id
                new_properties["type"] = node_type
                new_properties["created_at"] = timestamp
                new_properties["replaced_node_id"] = old_node_id
                new_properties["version"] = 1
                
                prop_str = ", ".join([f"{k}: ${k}" for k in new_properties.keys()])
                create_query = f"""
                CREATE (new:{node_type} {{{prop_str}}})
                WITH new
                MATCH (old {{id: $old_node_id, user_id: $user_id}})
                CREATE (new)-[:REPLACES]->(old)
                RETURN new.id as new_id
                """
                
                result = session.run(create_query, old_node_id=old_node_id, **new_properties)
                new_result = result.single()
                
                if new_result:
                    return {
                        "success": True,
                        "old_node_id": old_node_id,
                        "new_node_id": new_result.get("new_id"),
                        "strategy_used": "deprecate_and_create",
                        "message": f"Deprecated {node_type} {old_node_id}, created replacement {new_result.get('new_id')}"
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to create replacement node"
                    }
        
        except Exception as e:
            print(f"[UpdateHandler] Deprecate error: {e}")
            return {
                "success": False,
                "message": f"Deprecation error: {str(e)}"
            }
    
    def merge_duplicate_nodes(
        self,
        user_id: str,
        primary_node_id: str,
        duplicate_node_ids: List[str],
        node_type: str
    ) -> Dict[str, Any]:
        """
        Merge duplicate nodes into primary node.
        
        Strategy:
        1. Keep primary node
        2. Redirect all relationships from duplicates to primary
        3. Mark duplicates as merged
        4. Consolidate properties (prefer non-null, newer values)
        """
        if not self.driver or not duplicate_node_ids:
            return {"success": False}
        
        try:
            with self.driver.session() as session:
                merge_query = """
                MATCH (primary {id: $primary_id, user_id: $user_id})
                UNWIND $dup_ids as dup_id
                MATCH (dup {id: dup_id, user_id: $user_id, type: $node_type})
                
                // Redirect all relationships
                MATCH (dup)-[r]->(target)
                WHERE target.id <> $primary_id
                CREATE (primary)-[new_r:r.type]->(target)
                SET new_r = properties(r), new_r.source_merge = true
                DELETE r
                
                WITH primary, dup, count(r) as rel_count
                // Mark as merged
                SET dup.merged_into = $primary_id,
                    dup.merged_at = $timestamp,
                    dup.is_duplicate = true
                
                RETURN count(dup) as merged_count, sum(rel_count) as redirected_relations
                """
                
                result = session.run(
                    merge_query,
                    primary_id=primary_node_id,
                    user_id=user_id,
                    dup_ids=duplicate_node_ids,
                    node_type=node_type,
                    timestamp=datetime.utcnow().isoformat()
                )
                
                merge_result = result.single()
                
                if merge_result:
                    return {
                        "success": True,
                        "primary_node_id": primary_node_id,
                        "merged_count": merge_result.get("merged_count", 0),
                        "redirected_relations": merge_result.get("redirected_relations", 0),
                        "message": f"Merged {merge_result.get('merged_count')} duplicates into {primary_node_id}"
                    }
        
        except Exception as e:
            print(f"[UpdateHandler] Merge error: {e}")
            return {
                "success": False,
                "message": f"Merge error: {str(e)}"
            }
