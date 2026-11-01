//! intel_tex_2's prebuilt ISPC/ASTC objects were compiled from C++ and need libstdc++ at link time.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
