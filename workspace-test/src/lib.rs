// This crate exists to provide proper access to blockchain_node for integration tests
pub use blockchain_node;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
