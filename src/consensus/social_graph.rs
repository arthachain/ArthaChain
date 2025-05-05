use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct SocialGraphNode {
    pub id: String,
    pub reputation: f64,
    pub connections: Vec<String>,
    pub last_update: SystemTime,
}

impl SocialGraphNode {
    pub fn new(id: String) -> Self {
        Self {
            id,
            reputation: 0.0,
            connections: Vec::new(),
            last_update: SystemTime::now(),
        }
    }

    pub fn update_time(&mut self) {
        self.last_update = SystemTime::now();
    }

    pub fn add_connection(&mut self, node_id: String) {
        if !self.connections.contains(&node_id) {
            self.connections.push(node_id);
            self.update_time();
        }
    }

    pub fn remove_connection(&mut self, node_id: &str) {
        if let Some(pos) = self.connections.iter().position(|x| x == node_id) {
            self.connections.remove(pos);
            self.update_time();
        }
    }

    pub fn update_reputation(&mut self, new_reputation: f64) {
        self.reputation = new_reputation;
        self.update_time();
    }
} 