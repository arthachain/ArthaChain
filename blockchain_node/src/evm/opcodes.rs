use super::runtime::{EvmInterpreter, StepResult};
use super::types::{EvmAddress, EvmError, EvmLog};
use ethereum_types::{H256, U256};
use log::debug;
use sha3::{Digest, Keccak256};

impl<'a> EvmInterpreter<'a> {
    // Arithmetic operations
    pub fn op_add(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = a.overflowing_add(b).0; // Wrapping arithmetic
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_mul(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = a.overflowing_mul(b).0;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_sub(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = a.overflowing_sub(b).0;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_div(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = if b.is_zero() { U256::zero() } else { a / b };
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_sdiv(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;

        // Implement SDIV using two's complement without unary minus on U256
        if b.is_zero() {
            self.context.push_stack(U256::zero())?;
            return Ok(StepResult::Continue);
        }

        let a_neg = a.bit(255);
        let b_neg = b.bit(255);

        let a_mag = if a_neg {
            (!a).overflowing_add(U256::one()).0
        } else {
            a
        };
        let b_mag = if b_neg {
            (!b).overflowing_add(U256::one()).0
        } else {
            b
        };

        let mut q = if b_mag.is_zero() {
            U256::zero()
        } else {
            a_mag / b_mag
        };

        // If signs differ, negate the quotient using two's complement
        if a_neg ^ b_neg {
            q = (!q).overflowing_add(U256::one()).0;
        }

        self.context.push_stack(q)?;
        Ok(StepResult::Continue)
    }

    pub fn op_mod(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = if b.is_zero() { U256::zero() } else { a % b };
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_smod(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;

        let result = if b.is_zero() {
            U256::zero()
        } else {
            // Simplified signed modulo
            a % b
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_addmod(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(8)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let n = self.context.pop_stack()?;

        let result = if n.is_zero() {
            U256::zero()
        } else {
            (a + b) % n
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_mulmod(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(8)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let n = self.context.pop_stack()?;

        let result = if n.is_zero() {
            U256::zero()
        } else {
            (a * b) % n
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_exp(&mut self) -> Result<StepResult, EvmError> {
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;

        // Dynamic gas cost for exponentiation
        let gas_cost = 10 + (b.bits() as u64 / 8 + 1) * 50;
        self.context.consume_gas(gas_cost)?;

        let result = a.overflowing_pow(b).0;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_signextend(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5)?;
        let i = self.context.pop_stack()?;
        let x = self.context.pop_stack()?;

        let result = if i >= U256::from(32) {
            x
        } else {
            let i = i.as_u32();
            let bit_index = (8 * i + 7) as usize;
            if bit_index < 256 && x.bit(bit_index) {
                // Sign extend with 1s
                let mask = (U256::max_value() << (bit_index + 1)) & U256::max_value();
                x | mask
            } else {
                // Clear higher bits
                let mask = (U256::one() << (bit_index + 1)) - U256::one();
                x & mask
            }
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    // Comparison operations
    pub fn op_lt(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = if a < b { U256::one() } else { U256::zero() };
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_gt(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = if a > b { U256::one() } else { U256::zero() };
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_slt(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;

        // Signed comparison
        let a_negative = a.bit(255);
        let b_negative = b.bit(255);

        let result = match (a_negative, b_negative) {
            (true, false) => U256::one(),  // a < 0, b >= 0
            (false, true) => U256::zero(), // a >= 0, b < 0
            _ => {
                if a < b {
                    U256::one()
                } else {
                    U256::zero()
                }
            } // Same sign
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_sgt(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;

        // Signed comparison
        let a_negative = a.bit(255);
        let b_negative = b.bit(255);

        let result = match (a_negative, b_negative) {
            (false, true) => U256::one(),  // a >= 0, b < 0
            (true, false) => U256::zero(), // a < 0, b >= 0
            _ => {
                if a > b {
                    U256::one()
                } else {
                    U256::zero()
                }
            } // Same sign
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_eq(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = if a == b { U256::one() } else { U256::zero() };
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_iszero(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let result = if a.is_zero() {
            U256::one()
        } else {
            U256::zero()
        };
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    // Bitwise operations
    pub fn op_and(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = a & b;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_or(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = a | b;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_xor(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let b = self.context.pop_stack()?;
        let result = a ^ b;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_not(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let a = self.context.pop_stack()?;
        let result = !a;
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_byte(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let i = self.context.pop_stack()?;
        let x = self.context.pop_stack()?;

        let result = if i >= U256::from(32) {
            U256::zero()
        } else {
            let byte_index = i.as_usize();
            let shift = 8 * (31 - byte_index);
            (x >> shift) & U256::from(0xff)
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_shl(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let shift = self.context.pop_stack()?;
        let value = self.context.pop_stack()?;

        let result = if shift >= U256::from(256) {
            U256::zero()
        } else {
            value << shift.as_usize()
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_shr(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let shift = self.context.pop_stack()?;
        let value = self.context.pop_stack()?;

        let result = if shift >= U256::from(256) {
            U256::zero()
        } else {
            value >> shift.as_usize()
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    pub fn op_sar(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let shift = self.context.pop_stack()?;
        let value = self.context.pop_stack()?;

        let result = if shift >= U256::from(256) {
            // If value is negative, result is all 1s, else zero
            if value.bit(255) {
                !U256::zero()
            } else {
                U256::zero()
            }
        } else {
            // Arithmetic right shift (sign-extending)
            if value.bit(255) {
                // Negative number - fill with 1s
                let mask = !U256::zero() << (256 - shift.as_usize());
                (value >> shift.as_usize()) | mask
            } else {
                // Positive number - fill with 0s
                value >> shift.as_usize()
            }
        };

        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    // Keccak256
    pub fn op_keccak256(&mut self) -> Result<StepResult, EvmError> {
        let offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        // Dynamic gas cost
        let gas_cost = 30 + 6 * ((length + 31) / 32);
        self.context.consume_gas(gas_cost as u64)?;

        self.context.expand_memory(offset, length)?;

        let data = if length == 0 {
            Vec::new()
        } else {
            self.context.memory[offset..offset + length].to_vec()
        };

        let mut hasher = Keccak256::new();
        hasher.update(&data);
        let hash = hasher.finalize();

        let result = U256::from_big_endian(&hash);
        self.context.push_stack(result)?;
        Ok(StepResult::Continue)
    }

    // Environment information
    pub fn op_address(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let address = U256::from_big_endian(self.context.address.as_ref());
        self.context.push_stack(address)?;
        Ok(StepResult::Continue)
    }

    pub fn op_balance(&mut self) -> Result<StepResult, EvmError> {
        let address_u256 = self.context.pop_stack()?;
        self.context.consume_gas(700)?; // High gas cost for external state access

        let mut address_bytes = [0u8; 20];
        address_u256.to_big_endian(&mut address_bytes[..]);
        let address = EvmAddress::from(address_bytes);

        // Get account balance
        let balance = match self.backend.get_account(&address) {
            Ok(account) => account.balance,
            Err(_) => U256::zero(),
        };

        self.context.push_stack(balance)?;
        Ok(StepResult::Continue)
    }

    pub fn op_origin(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        // In a real implementation, this would track the original transaction sender
        let origin = U256::from_big_endian(self.context.caller.as_ref());
        self.context.push_stack(origin)?;
        Ok(StepResult::Continue)
    }

    pub fn op_caller(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let caller = U256::from_big_endian(self.context.caller.as_ref());
        self.context.push_stack(caller)?;
        Ok(StepResult::Continue)
    }

    pub fn op_callvalue(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        self.context.push_stack(self.context.value)?;
        Ok(StepResult::Continue)
    }

    pub fn op_calldataload(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let offset = self.context.pop_stack()?.as_usize();

        let mut result = [0u8; 32];
        for i in 0..32 {
            if offset + i < self.context.data.len() {
                result[i] = self.context.data[offset + i];
            }
        }

        let value = U256::from_big_endian(&result);
        self.context.push_stack(value)?;
        Ok(StepResult::Continue)
    }

    pub fn op_calldatasize(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let size = U256::from(self.context.data.len());
        self.context.push_stack(size)?;
        Ok(StepResult::Continue)
    }

    pub fn op_calldatacopy(&mut self) -> Result<StepResult, EvmError> {
        let dest_offset = self.context.pop_stack()?.as_usize();
        let data_offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        let gas_cost = 3 + 3 * ((length + 31) / 32);
        self.context.consume_gas(gas_cost as u64)?;

        self.context.expand_memory(dest_offset, length)?;

        for i in 0..length {
            let byte = if data_offset + i < self.context.data.len() {
                self.context.data[data_offset + i]
            } else {
                0
            };
            self.context.memory[dest_offset + i] = byte;
        }

        Ok(StepResult::Continue)
    }

    pub fn op_codesize(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let size = U256::from(self.context.code.len());
        self.context.push_stack(size)?;
        Ok(StepResult::Continue)
    }

    pub fn op_codecopy(&mut self) -> Result<StepResult, EvmError> {
        let dest_offset = self.context.pop_stack()?.as_usize();
        let code_offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        let gas_cost = 3 + 3 * ((length + 31) / 32);
        self.context.consume_gas(gas_cost as u64)?;

        self.context.expand_memory(dest_offset, length)?;

        for i in 0..length {
            let byte = if code_offset + i < self.context.code.len() {
                self.context.code[code_offset + i]
            } else {
                0
            };
            self.context.memory[dest_offset + i] = byte;
        }

        Ok(StepResult::Continue)
    }

    pub fn op_gasprice(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        // Default gas price
        let gas_price = U256::from(20_000_000_000u64); // 20 gwei
        self.context.push_stack(gas_price)?;
        Ok(StepResult::Continue)
    }

    pub fn op_extcodesize(&mut self) -> Result<StepResult, EvmError> {
        let address_u256 = self.context.pop_stack()?;
        self.context.consume_gas(700)?;

        let mut address_bytes = [0u8; 20];
        address_u256.to_big_endian(&mut address_bytes[..]);
        let address = EvmAddress::from(address_bytes);

        let code_size = match self.backend.get_code(&address) {
            Ok(code) => code.len(),
            Err(_) => 0,
        };

        self.context.push_stack(U256::from(code_size))?;
        Ok(StepResult::Continue)
    }

    pub fn op_extcodecopy(&mut self) -> Result<StepResult, EvmError> {
        let address_u256 = self.context.pop_stack()?;
        let dest_offset = self.context.pop_stack()?.as_usize();
        let code_offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        let gas_cost = 700 + 3 * ((length + 31) / 32);
        self.context.consume_gas(gas_cost as u64)?;

        let mut address_bytes = [0u8; 20];
        address_u256.to_big_endian(&mut address_bytes[..]);
        let address = EvmAddress::from(address_bytes);

        self.context.expand_memory(dest_offset, length)?;

        let code = self.backend.get_code(&address).unwrap_or_default();

        for i in 0..length {
            let byte = if code_offset + i < code.len() {
                code[code_offset + i]
            } else {
                0
            };
            self.context.memory[dest_offset + i] = byte;
        }

        Ok(StepResult::Continue)
    }

    pub fn op_returndatasize(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let size = U256::from(self.context.return_data.len());
        self.context.push_stack(size)?;
        Ok(StepResult::Continue)
    }

    pub fn op_returndatacopy(&mut self) -> Result<StepResult, EvmError> {
        let dest_offset = self.context.pop_stack()?.as_usize();
        let data_offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        let gas_cost = 3 + 3 * ((length + 31) / 32);
        self.context.consume_gas(gas_cost as u64)?;

        if data_offset + length > self.context.return_data.len() {
            return Err(EvmError::InvalidTransaction(
                "Return data out of bounds".to_string(),
            ));
        }

        self.context.expand_memory(dest_offset, length)?;

        for i in 0..length {
            self.context.memory[dest_offset + i] = self.context.return_data[data_offset + i];
        }

        Ok(StepResult::Continue)
    }

    // Block information
    pub fn op_blockhash(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(20)?;
        let block_number = self.context.pop_stack()?;

        // Simplified block hash (in real implementation would query blockchain)
        let current_block = U256::from(self.context.block_number);
        let hash =
            if block_number >= current_block || current_block - block_number > U256::from(256) {
                U256::zero()
            } else {
                // Generate a deterministic hash based on block number
                let mut hasher = Keccak256::new();
                hasher.update(b"block_hash");
                hasher.update(&block_number.as_u64().to_be_bytes());
                let hash = hasher.finalize();
                U256::from_big_endian(&hash)
            };

        self.context.push_stack(hash)?;
        Ok(StepResult::Continue)
    }

    pub fn op_coinbase(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        // Default coinbase address
        let coinbase = U256::zero();
        self.context.push_stack(coinbase)?;
        Ok(StepResult::Continue)
    }

    pub fn op_timestamp(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let timestamp = U256::from(self.context.block_timestamp);
        self.context.push_stack(timestamp)?;
        Ok(StepResult::Continue)
    }

    pub fn op_number(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let block_number = U256::from(self.context.block_number);
        self.context.push_stack(block_number)?;
        Ok(StepResult::Continue)
    }

    pub fn op_difficulty(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        // Default difficulty
        let difficulty = U256::from(2000000u64);
        self.context.push_stack(difficulty)?;
        Ok(StepResult::Continue)
    }

    pub fn op_gaslimit(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let gas_limit = U256::from(self.context.gas_limit);
        self.context.push_stack(gas_limit)?;
        Ok(StepResult::Continue)
    }

    // Storage and memory operations
    pub fn op_pop(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        self.context.pop_stack()?;
        Ok(StepResult::Continue)
    }

    pub fn op_mload(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let offset = self.context.pop_stack()?.as_usize();

        self.context.expand_memory(offset, 32)?;

        let mut word = [0u8; 32];
        for i in 0..32 {
            if offset + i < self.context.memory.len() {
                word[i] = self.context.memory[offset + i];
            }
        }

        let value = U256::from_big_endian(&word);
        self.context.push_stack(value)?;
        Ok(StepResult::Continue)
    }

    pub fn op_mstore(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let offset = self.context.pop_stack()?.as_usize();
        let value = self.context.pop_stack()?;

        self.context.expand_memory(offset, 32)?;

        let mut word = [0u8; 32];
        value.to_big_endian(&mut word);

        for i in 0..32 {
            self.context.memory[offset + i] = word[i];
        }

        Ok(StepResult::Continue)
    }

    pub fn op_mstore8(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let offset = self.context.pop_stack()?.as_usize();
        let value = self.context.pop_stack()?;

        self.context.expand_memory(offset, 1)?;

        self.context.memory[offset] = (value.as_u64() & 0xff) as u8;

        Ok(StepResult::Continue)
    }

    pub async fn op_sload(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(800)?;
        let key_u256 = self.context.pop_stack()?;

        let mut key_bytes = [0u8; 32];
        key_u256.to_big_endian(&mut key_bytes);
        let key = H256::from(key_bytes);

        // Check local storage first
        let value = if let Some(&local_value) = self.context.storage.get(&key) {
            local_value
        } else {
            // Load from backend storage
            self.backend
                .get_storage(&self.context.address, key)
                .unwrap_or(H256::zero())
        };

        let value_u256 = U256::from_big_endian(value.as_ref());
        self.context.push_stack(value_u256)?;
        Ok(StepResult::Continue)
    }

    pub async fn op_sstore(&mut self) -> Result<StepResult, EvmError> {
        let key_u256 = self.context.pop_stack()?;
        let value_u256 = self.context.pop_stack()?;

        let mut key_bytes = [0u8; 32];
        key_u256.to_big_endian(&mut key_bytes);
        let key = H256::from(key_bytes);

        let mut value_bytes = [0u8; 32];
        value_u256.to_big_endian(&mut value_bytes);
        let value = H256::from(value_bytes);

        // Check current value for gas calculation
        let current_value = if let Some(&local_value) = self.context.storage.get(&key) {
            local_value
        } else {
            self.backend
                .get_storage(&self.context.address, key)
                .unwrap_or(H256::zero())
        };

        let gas_cost = if current_value == H256::zero() && value != H256::zero() {
            20000 // Setting storage from zero
        } else {
            5000 // Modifying existing storage
        };

        self.context.consume_gas(gas_cost)?;

        // Store in local storage (will be committed later)
        self.context.storage.insert(key, value);

        Ok(StepResult::Continue)
    }

    pub fn op_jump(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(8)?;
        let dest = self.context.pop_stack()?.as_usize();

        if dest >= self.context.code.len() {
            return Err(EvmError::InvalidJumpDestination);
        }

        // Check if destination is a valid JUMPDEST
        if self.context.code[dest] != 0x5b {
            return Err(EvmError::InvalidJumpDestination);
        }

        self.context.pc = dest;
        Ok(StepResult::Continue)
    }

    pub fn op_jumpi(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(10)?;
        let dest = self.context.pop_stack()?.as_usize();
        let condition = self.context.pop_stack()?;

        if !condition.is_zero() {
            if dest >= self.context.code.len() {
                return Err(EvmError::InvalidJumpDestination);
            }

            // Check if destination is a valid JUMPDEST
            if self.context.code[dest] != 0x5b {
                return Err(EvmError::InvalidJumpDestination);
            }

            self.context.pc = dest;
        }

        Ok(StepResult::Continue)
    }

    pub fn op_pc(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let pc = U256::from(self.context.pc - 1); // PC before increment
        self.context.push_stack(pc)?;
        Ok(StepResult::Continue)
    }

    pub fn op_msize(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let size = U256::from(self.context.memory.len());
        self.context.push_stack(size)?;
        Ok(StepResult::Continue)
    }

    pub fn op_gas(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(2)?;
        let remaining_gas = U256::from(self.context.gas_limit - self.context.gas_used);
        self.context.push_stack(remaining_gas)?;
        Ok(StepResult::Continue)
    }

    pub fn op_jumpdest(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(1)?;
        // JUMPDEST is just a marker, no operation needed
        Ok(StepResult::Continue)
    }

    // Push operations
    pub fn op_push(&mut self, n: u8) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;

        let mut data = vec![0u8; n as usize];
        for i in 0..n as usize {
            if self.context.pc + i < self.context.code.len() {
                data[i] = self.context.code[self.context.pc + i];
            }
        }

        self.context.pc += n as usize;

        let value = U256::from_big_endian(&data);
        self.context.push_stack(value)?;
        Ok(StepResult::Continue)
    }

    // Duplicate operations
    pub fn op_dup(&mut self, n: u8) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let value = self.context.peek_stack((n - 1) as usize)?;
        self.context.push_stack(value)?;
        Ok(StepResult::Continue)
    }

    // Swap operations
    pub fn op_swap(&mut self, n: u8) -> Result<StepResult, EvmError> {
        self.context.consume_gas(3)?;
        let stack_len = self.context.stack.len();

        if stack_len < (n + 1) as usize {
            return Err(EvmError::StackUnderflow);
        }

        let top_idx = stack_len - 1;
        let swap_idx = stack_len - 1 - n as usize;

        self.context.stack.swap(top_idx, swap_idx);
        Ok(StepResult::Continue)
    }

    // Logging operations
    pub fn op_log(&mut self, n: u8) -> Result<StepResult, EvmError> {
        let offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        let gas_cost = 375 + 8 * length + 375 * n as usize;
        self.context.consume_gas(gas_cost as u64)?;

        self.context.expand_memory(offset, length)?;

        let mut topics = Vec::new();
        for _ in 0..n {
            let topic_u256 = self.context.pop_stack()?;
            let mut topic_bytes = [0u8; 32];
            topic_u256.to_big_endian(&mut topic_bytes);
            topics.push(H256::from(topic_bytes));
        }

        let data = if length == 0 {
            Vec::new()
        } else {
            self.context.memory[offset..offset + length].to_vec()
        };

        let log = EvmLog {
            address: self.context.address,
            topics,
            data,
        };

        self.context.logs.push(log);
        Ok(StepResult::Continue)
    }

    // System operations (simplified implementations)
    pub async fn op_create(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(32000)?;
        // Simplified CREATE - would need full contract creation logic
        self.context.push_stack(U256::zero())?; // Return zero address for now
        Ok(StepResult::Continue)
    }

    pub async fn op_call(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(700)?;
        // Simplified CALL - would need full call logic
        self.context.push_stack(U256::one())?; // Return success for now
        Ok(StepResult::Continue)
    }

    pub async fn op_callcode(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(700)?;
        // Simplified CALLCODE
        self.context.push_stack(U256::one())?;
        Ok(StepResult::Continue)
    }

    pub fn op_return(&mut self) -> Result<StepResult, EvmError> {
        let offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        self.context.expand_memory(offset, length)?;

        let return_data = if length == 0 {
            Vec::new()
        } else {
            self.context.memory[offset..offset + length].to_vec()
        };

        Ok(StepResult::Return(return_data))
    }

    pub async fn op_delegatecall(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(700)?;
        // Simplified DELEGATECALL
        self.context.push_stack(U256::one())?;
        Ok(StepResult::Continue)
    }

    pub async fn op_create2(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(32000)?;
        // Simplified CREATE2
        self.context.push_stack(U256::zero())?;
        Ok(StepResult::Continue)
    }

    pub async fn op_staticcall(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(700)?;
        // Simplified STATICCALL
        self.context.push_stack(U256::one())?;
        Ok(StepResult::Continue)
    }

    pub fn op_revert(&mut self) -> Result<StepResult, EvmError> {
        let offset = self.context.pop_stack()?.as_usize();
        let length = self.context.pop_stack()?.as_usize();

        self.context.expand_memory(offset, length)?;

        let revert_data = if length == 0 {
            Vec::new()
        } else {
            self.context.memory[offset..offset + length].to_vec()
        };

        Ok(StepResult::Revert(revert_data))
    }

    pub fn op_invalid(&mut self) -> Result<StepResult, EvmError> {
        Err(EvmError::InvalidOpcode(0xfe))
    }

    pub async fn op_selfdestruct(&mut self) -> Result<StepResult, EvmError> {
        self.context.consume_gas(5000)?;
        // Simplified SELFDESTRUCT
        self.finished = true;
        Ok(StepResult::Return(Vec::new()))
    }
}
