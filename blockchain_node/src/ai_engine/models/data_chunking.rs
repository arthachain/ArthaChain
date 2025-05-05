use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use log::{info, warn, debug};

/// Adaptive data chunking model using NumPy
pub struct AdaptiveChunker {
    /// Python interpreter
    py_interpreter: Python<'static>,
    /// NumPy module
    numpy: PyObject,
    /// Chunking parameters
    params: ChunkingParams,
    /// Chunk statistics
    stats: ChunkStats,
}

/// Chunking parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingParams {
    /// Minimum chunk size
    pub min_size: usize,
    /// Maximum chunk size
    pub max_size: usize,
    /// Target compression ratio
    pub target_compression: f32,
    /// Similarity threshold
    pub similarity_threshold: f32,
    /// Adaptation rate
    pub adaptation_rate: f32,
}

/// Chunk statistics
#[derive(Debug, Clone, Default)]
pub struct ChunkStats {
    /// Average chunk size
    pub avg_size: f32,
    /// Compression ratios
    pub compression_ratios: Vec<f32>,
    /// Boundary frequencies
    pub boundary_freq: HashMap<usize, usize>,
    /// Content entropy
    pub content_entropy: Vec<f32>,
}

impl AdaptiveChunker {
    /// Create a new adaptive chunker
    pub fn new(params: ChunkingParams) -> Result<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Import NumPy
        let numpy = py.import("numpy")?;

        Ok(Self {
            py_interpreter: py,
            numpy: numpy.into(),
            params,
            stats: ChunkStats::default(),
        })
    }

    /// Chunk data adaptively
    pub fn chunk_data(&mut self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Convert data to NumPy array
        let data_array = PyArray1::from_slice(py, data)?;

        // Calculate rolling hash
        let window_size = 16;
        let rolling_hash = self.calculate_rolling_hash(py, data_array, window_size)?;

        // Find chunk boundaries
        let boundaries = self.find_chunk_boundaries(py, &rolling_hash)?;

        // Create chunks
        let mut chunks = Vec::new();
        let mut start = 0;

        for &end in &boundaries {
            if end - start >= self.params.min_size && end - start <= self.params.max_size {
                chunks.push(data[start..end].to_vec());
                start = end;
            }
        }

        // Add final chunk if needed
        if start < data.len() {
            chunks.push(data[start..].to_vec());
        }

        // Update statistics
        self.update_stats(&chunks)?;

        // Adapt parameters if needed
        self.adapt_parameters()?;

        Ok(chunks)
    }

    /// Calculate rolling hash using NumPy
    fn calculate_rolling_hash(
        &self,
        py: Python,
        data: &PyArray1<u8>,
        window_size: usize,
    ) -> Result<Vec<u32>> {
        let code = format!(
            r#"
def rolling_hash(data, window_size):
    # Rabin-Karp rolling hash
    prime = 31
    mod_val = 1 << 32
    
    # Calculate initial hash
    hash_val = 0
    for i in range(window_size):
        hash_val = (hash_val * prime + int(data[i])) % mod_val
    
    hashes = [hash_val]
    
    # Calculate rolling hash for remaining windows
    for i in range(len(data) - window_size):
        hash_val = (
            (hash_val - int(data[i]) * pow(prime, window_size - 1, mod_val)) * prime +
            int(data[i + window_size])
        ) % mod_val
        hashes.append(hash_val)
    
    return np.array(hashes, dtype=np.uint32)

result = rolling_hash(data, {})
            "#,
            window_size
        );

        let locals = PyDict::new(py);
        locals.set_item("data", data)?;
        locals.set_item("np", self.numpy.as_ref(py))?;

        let result = py.eval(&code, None, Some(locals))?;
        let hashes = result.extract::<Vec<u32>>()?;

        Ok(hashes)
    }

    /// Find chunk boundaries using content-defined chunking
    fn find_chunk_boundaries(&self, py: Python, rolling_hash: &[u32]) -> Result<Vec<usize>> {
        let hash_array = PyArray1::from_slice(py, rolling_hash)?;

        let code = format!(
            r#"
def find_boundaries(hashes, min_size, max_size, threshold):
    # Find local maxima in rolling hash values
    window = min_size // 2
    maxima = np.zeros_like(hashes, dtype=bool)
    
    for i in range(window, len(hashes) - window):
        if all(hashes[i] >= hashes[i-window:i]) and \
           all(hashes[i] >= hashes[i+1:i+window+1]):
            maxima[i] = True
    
    # Filter boundaries based on threshold
    boundaries = np.where(
        (maxima) & (hashes > np.mean(hashes) * threshold)
    )[0]
    
    # Ensure chunk size constraints
    valid_boundaries = []
    last_boundary = 0
    
    for b in boundaries:
        size = b - last_boundary
        if size >= min_size and size <= max_size:
            valid_boundaries.append(b)
            last_boundary = b
    
    return np.array(valid_boundaries)

result = find_boundaries(
    hashes,
    {},  # min_size
    {},  # max_size
    {}   # threshold
)
            "#,
            self.params.min_size,
            self.params.max_size,
            self.params.similarity_threshold
        );

        let locals = PyDict::new(py);
        locals.set_item("hashes", hash_array)?;
        locals.set_item("np", self.numpy.as_ref(py))?;

        let result = py.eval(&code, None, Some(locals))?;
        let boundaries = result.extract::<Vec<usize>>()?;

        Ok(boundaries)
    }

    /// Update chunking statistics
    fn update_stats(&mut self, chunks: &[Vec<u8>]) -> Result<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Calculate average chunk size
        let sizes: Vec<_> = chunks.iter().map(|c| c.len()).collect();
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        self.stats.avg_size = avg_size;

        // Update boundary frequencies
        for size in sizes {
            *self.stats.boundary_freq.entry(size).or_insert(0) += 1;
        }

        // Calculate compression ratios and entropy
        for chunk in chunks {
            let chunk_array = PyArray1::from_slice(py, chunk)?;
            
            // Compression ratio using numpy's unique count
            let unique_count = py
                .eval(
                    "len(np.unique(chunk))",
                    None,
                    Some([("chunk", chunk_array)].into_py_dict(py))
                )?
                .extract::<usize>()?;
            
            let ratio = unique_count as f32 / chunk.len() as f32;
            self.stats.compression_ratios.push(ratio);

            // Calculate entropy
            let entropy = py
                .eval(
                    r#"
                    hist = np.bincount(chunk) / len(chunk)
                    -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
                    "#,
                    None,
                    Some([("chunk", chunk_array)].into_py_dict(py))
                )?
                .extract::<f32>()?;
            
            self.stats.content_entropy.push(entropy);
        }

        Ok(())
    }

    /// Adapt chunking parameters based on statistics
    fn adapt_parameters(&mut self) -> Result<()> {
        // Calculate average compression ratio
        let avg_compression = self.stats.compression_ratios.iter().sum::<f32>() /
            self.stats.compression_ratios.len() as f32;

        // Adjust similarity threshold based on compression target
        if avg_compression > self.params.target_compression {
            self.params.similarity_threshold *= 1.0 + self.params.adaptation_rate;
        } else {
            self.params.similarity_threshold *= 1.0 - self.params.adaptation_rate;
        }

        // Clamp threshold to reasonable range
        self.params.similarity_threshold = self.params.similarity_threshold
            .max(0.1)
            .min(0.9);

        // Adjust chunk size bounds based on entropy
        let avg_entropy = self.stats.content_entropy.iter().sum::<f32>() /
            self.stats.content_entropy.len() as f32;

        if avg_entropy > 0.7 { // High entropy -> larger chunks
            self.params.min_size = (self.params.min_size as f32 * 1.1) as usize;
            self.params.max_size = (self.params.max_size as f32 * 1.1) as usize;
        } else { // Low entropy -> smaller chunks
            self.params.min_size = (self.params.min_size as f32 * 0.9) as usize;
            self.params.max_size = (self.params.max_size as f32 * 0.9) as usize;
        }

        // Clamp size bounds
        self.params.min_size = self.params.min_size.max(64).min(4096);
        self.params.max_size = self.params.max_size.max(4096).min(65536);

        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &ChunkStats {
        &self.stats
    }

    /// Get current parameters
    pub fn get_params(&self) -> &ChunkingParams {
        &self.params
    }
} 