use env_logger::Builder;
use log::{LevelFilter, info, error};
use std::io::Write;
use std::path::Path;
use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};

/// Initialize the logger with file and console output
pub fn init_logger(log_file: Option<&Path>, level: LevelFilter) -> Result<()> {
    let mut builder = Builder::new();
    
    // Set default log level
    builder.filter_level(level);
    
    // Format with timestamp, level, and target
    builder.format(|buf, record| {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        writeln!(
            buf,
            "[{} {} {}] {}",
            timestamp,
            record.level(),
            record.target(),
            record.args()
        )
    });
    
    // Set up file logging if requested
    if let Some(log_path) = log_file {
        // Ensure parent directory exists
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create log directory")?;
        }
        
        // Open log file with append mode
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .context("Failed to open log file")?;
        
        // Create a second writer that writes to both stderr and the file
        builder.target(env_logger::Target::Pipe(Box::new(FileAndStderr {
            file,
        })));
    }
    
    // Initialize the logger
    builder.init();
    
    info!("Logger initialized with level {}", level);
    
    Ok(())
}

/// Parse a log level string
pub fn parse_log_level(level: &str) -> LevelFilter {
    match level.to_lowercase().as_str() {
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "info" => LevelFilter::Info,
        "warn" => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        "off" => LevelFilter::Off,
        _ => {
            error!("Invalid log level: {}, using info", level);
            LevelFilter::Info
        }
    }
}

/// Custom log writer that outputs to both a file and stderr
struct FileAndStderr {
    file: File,
}

impl Write for FileAndStderr {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Write to stderr
        std::io::stderr().write_all(buf)?;
        
        // Write to file
        self.file.write_all(buf)?;
        
        Ok(buf.len())
    }
    
    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()?;
        self.file.flush()?;
        Ok(())
    }
}

/// Log a startup banner with version and configuration info
pub fn log_startup_banner(version: &str, config_summary: &str) {
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("                  Artha Chain Node v{}", version);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("Configuration:");
    for line in config_summary.lines() {
        info!("  {}", line);
    }
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
} 