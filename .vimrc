map <F5> :wa<CR> :! CARGO_INCREMENTAL=1 RUST_BACKTRACE=1 cargo run<CR>
map <C-F5> :wa<CR> :! CARGO_INCREMENTAL=1 cargo run --release<CR>
map <C-S-F5> :wa<CR> :! CARGO_INCREMENTAL=1 cargo test -p lsys<CR>
