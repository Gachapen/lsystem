map <F5> :wa<CR> :T CARGO_INCREMENTAL=1 cargo build && rust-gdb -ex "set substitute-path /buildslave/rust-buildbot/slave/nightly-dist-rustc-linux/build/src $RUST_SRC_PATH" -x .gdbrun target/debug/lsystem<CR>
map <F29> :wa<CR> :T CARGO_INCREMENTAL=1 RUST_BACKTRACE=1 cargo run --release<CR>
map <F6> :wa<CR> :T CARGO_INCREMENTAL=1 RUST_BACKTRACE=1 cargo test -p lsys -p abnf -p lsystem<CR>
