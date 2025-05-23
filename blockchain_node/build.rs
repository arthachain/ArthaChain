fn main() {
    // Skip building the problem files in tests for now
    println!("cargo:rustc-cfg=skip_problematic_modules");

    // Add rustc-check-cfg to handle the unexpected cfg check
    println!("cargo:rustc-check-cfg=cfg(skip_problematic_modules)");
}
