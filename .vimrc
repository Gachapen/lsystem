map <F5> :wa<CR> :! RUST_BACKTRACE=1 cargo run<CR>
map <C-F5> :wa<CR> :! cargo run --release<CR>
map <C-S-F5> :wa<CR> :! cargo test -p lsys<CR>
