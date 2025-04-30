impl Blockchain {
    pub async fn get_difficulty(&self) -> Result<u64, anyhow::Error> {
        let state = self.state.lock().await;
        Ok(state.get_difficulty())
    }

    pub async fn get_total_transactions(&self) -> Result<u64, anyhow::Error> {
        let state = self.state.lock().await;
        Ok(state.get_total_transactions())
    }
} 