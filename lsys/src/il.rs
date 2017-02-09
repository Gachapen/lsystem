use std::fmt;
use std::result::Result;
use std::option::Option;

use common::Instruction;
use common::CommandMap;
use common::create_command_map;
use common::map_word_to_instructions;
use common::Rewriter;

fn matches_left_context(word_left: &str, context: u8, ignores: &Vec<u8>) -> bool {
    let bytes = word_left.as_bytes();

    let mut i = word_left.len();
    let mut found_match = false;
    let mut searching = true;
    let mut branching = 0usize;

    while searching && i > 0 {
        i -= 1;

        while searching && bytes[i] == '[' as u8 {
            if branching > 0 {
                branching -= 1;
            }

            if i > 0 {
                i -= 1;
            } else {
                searching = false;
            }
        }

        while searching && bytes[i] == ']' as u8 {
            branching += 1;

            if i > 0 {
                i -= 1;
            } else {
                searching = false;
            }
        }

        if searching && branching == 0 && !ignores.iter().any(|&c| c == bytes[i]) {
            searching = false;

            if bytes[i] == context {
                found_match = true;
            }
        }
    }

    found_match
}

// Old version that accepts right context on side branches.
//fn matches_right_context(word_right: &str, context: u8, ignores: &Vec<u8>) -> bool {
//    let bytes = word_right.as_bytes();
//    let length = word_right.len();

//    let mut i = 0usize;
//    let mut searching = true;
//    let mut found_match = false;

//    let mut branch_stack = Vec::<bool>::new();
//    let mut safe = true;

//    while searching && i < length {
//        // Parse branch popping.
//        while i < length && bytes[i] == ']' as u8 {
//            if branch_stack.is_empty() {
//                // Popped out of context. Match not possible.
//                searching = false;
//            } else {
//                // Return to last branching point.
//                safe = branch_stack.pop().expect("Branching error");
//            }

//            i += 1;
//        }

//        if i < length && searching {
//            // Parse branch pushing.
//            while i < length && bytes[i] == '[' as u8 {
//                // Store branching point.
//                branch_stack.push(safe);
//                i += 1;
//            }

//            // This branch is safe and the current letter can not be ignored.
//            if i < length && safe && !ignores.iter().any(|&c| c == bytes[i]) {
//                if bytes[i] == context {
//                    found_match = true;
//                    searching = false;
//                } else {
//                    // This branch is not longer safe.
//                    safe = false;
//                }
//            }

//            i += 1;
//        }
//    }

//    found_match
//}

fn matches_right_context(word_right: &str, context: u8, ignores: &Vec<u8>) -> bool {
    let bytes = word_right.as_bytes();
    let length = word_right.len();

    let mut i = 0usize;
    let mut searching = true;
    let mut found_match = false;
    let mut branching = 0usize;

    while searching && i < length {
        // Parse branch popping.
        while i < length && bytes[i] == ']' as u8 {
            if branching == 0 {
                // Popped out of context. Match not possible.
                searching = false;
            } else {
                // Return to last branching point.
                branching -= 1;
            }

            i += 1;
        }

        if i < length && searching {

            // Parse branch pushing.
            while i < length && bytes[i] == '[' as u8 {
                // Store branching point.
                branching += 1;
                i += 1;
            }

            // This branch is safe and the current letter can not be ignored.
            if i < length && branching == 0 && !ignores.iter().any(|&c| c == bytes[i]) {
                searching = false;
                if bytes[i] == context {
                    found_match = true;
                }
            }

            i += 1;
        }
    }

    found_match
}

fn matches_predecessor(word: &str, i: usize, pred: &Predecessor, ignores: &Vec<u8>) -> bool {
    let bytes = word.as_bytes();

    if pred.strict != bytes[i] {
        return false;
    }

    if let Some(left) = pred.left {
        if i <= 0 {
            return false;
        }

        if !matches_left_context(&word[0..i], left, &ignores) {
            return false;
        }
    }

    if let Some(right) = pred.right {
        if i >= bytes.len() - 1 {
            return false;
        }

        if !matches_right_context(&word[i+1..], right, &ignores) {
            return false;
        }
    }

    //println!("Match with {:?} on {} at {}", pred, word, i);
    true
}

fn expand_lsystem(axiom: &str, rules: &Vec<Production>, iterations: u32, ignore: &Vec<u8>) -> String {
    let mut lword = String::from(axiom);
    //println!("0: {}", lword);

    for _ in 0..iterations {
        let mut expanded_lword = String::with_capacity(lword.len());

        for (i, lchar) in lword.as_bytes().iter().cloned().enumerate() {
            let prod = rules.iter().find(|&prod| {
                matches_predecessor(&lword, i, &prod.predecessor, &ignore)
            });

            if let Some(prod) = prod {
                expanded_lword.push_str(&prod.successor.clone());
            } else {
                expanded_lword.push(lchar as char);
            }
        }
        lword = expanded_lword;

        //println!("{}: {}", i+1, lword);
    }

    lword
}

pub struct Predecessor {
    pub strict: u8,
    pub left: Option<u8>,
    pub right: Option<u8>,
}

impl Predecessor {
    pub fn without_context(strict: char) -> Predecessor {
        Predecessor {
            strict: strict as u8,
            left: None,
            right: None,
        }
    }

    pub fn with_left_context(left: char, strict: char) -> Predecessor {
        Predecessor {
            strict: strict as u8,
            left: Some(left as u8),
            right: None,
        }
    }

    pub fn with_right_context(strict: char, right: char) -> Predecessor {
        Predecessor {
            strict: strict as u8,
            left: None,
            right: Some(right as u8),
        }
    }

    pub fn with_context(left: char, strict: char, right: char) -> Predecessor {
        Predecessor {
            strict: strict as u8,
            left: Some(left as u8),
            right: Some(right as u8),
        }
    }
}

impl fmt::Debug for Predecessor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(left) = self.left {
            let result = write!(f, "{} < ", left as char);
            if result.is_err() {
                return result;
            }
        }

        let result = write!(f, "{}", self.strict as char);
        if result.is_err() {
            return result;
        }

        if let Some(right) = self.right {
            let result = write!(f, " > {}", right as char);
            if result.is_err() {
                return result;
            }
        }

        Result::Ok::<(), fmt::Error>(())
    }
}

pub struct Production {
    pub predecessor: Predecessor,
    pub successor: String,
}

impl Production {
    pub fn without_context(strict: char, successor: &str) -> Production {
        Production {
            predecessor: Predecessor::without_context(strict),
            successor: successor.to_string(),
        }
    }

    pub fn with_left_context(left: char, strict: char, successor: &str) -> Production {
        Production {
            predecessor: Predecessor::with_left_context(left, strict),
            successor: successor.to_string(),
        }
    }

    pub fn with_right_context(strict: char, right: char, successor: &str) -> Production {
        Production {
            predecessor: Predecessor::with_right_context(strict, right),
            successor: successor.to_string(),
        }
    }

    pub fn with_context(left: char, strict: char, right: char, successor: &str) -> Production {
        Production {
            predecessor: Predecessor::with_context(left, strict, right),
            successor: successor.to_string(),
        }
    }
}

pub struct LSystem {
    pub command_map: CommandMap,
    pub productions: Vec<Production>,
    pub axiom: String,
    ignore: Vec<u8>,
}

impl LSystem {
    pub fn new() -> LSystem {
        LSystem {
            command_map: create_command_map(),
            productions: Vec::new(),
            axiom: String::new(),
            ignore: Vec::new(),
        }
    }

    pub fn ignore_from_context(&mut self, ignore: &str) {
        self.ignore.reserve(ignore.as_bytes().len());
        for c in ignore.bytes() {
            self.ignore.push(c as u8);
        }
    }
}

impl Rewriter for LSystem {
    fn instructions(&self, iterations: u32) -> Vec<Instruction> {
        let lword = expand_lsystem(&self.axiom, &self.productions, iterations, &self.ignore);
        map_word_to_instructions(&lword, &self.command_map)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_left_context_test_single() {
        assert!(super::matches_left_context("A", 'A' as u8, &vec![]));
        assert!(!super::matches_left_context("B", 'A' as u8, &vec![]));
    }

    #[test]
    fn matches_left_context_test_ignore() {
        assert!(!super::matches_left_context("AB", 'A' as u8, &vec![]));
        assert!(super::matches_left_context("AB", 'A' as u8, &vec!['B' as u8]));
        assert!(super::matches_left_context("ABC", 'A' as u8, &vec!['B' as u8, 'C' as u8]));
    }

    #[test]
    fn matches_left_context_test_branch() {
        assert!(super::matches_left_context("A[B]", 'A' as u8, &vec![]));
        assert!(super::matches_left_context("A[[B]C][D[E]]", 'A' as u8, &vec![]));
        assert!(super::matches_left_context("[[A[[B]C][[[D[E]][", 'A' as u8, &vec![]));
    }

    #[test]
    fn matches_left_context_test_branch_ignore() {
        assert!(super::matches_left_context("AF[[B]C]F[D[E]]F", 'A' as u8, &vec!['C' as u8, 'F' as u8]));
    }

    #[test]
    fn matches_right_context_test_single() {
        assert!(super::matches_right_context("A", 'A' as u8, &vec![]));
        assert!(!super::matches_right_context("B", 'A' as u8, &vec![]));
    }

    #[test]
    fn matches_right_context_test_ignore() {
        assert!(!super::matches_right_context("BA", 'A' as u8, &vec![]));
        assert!(super::matches_right_context("BA", 'A' as u8, &vec!['B' as u8]));
        assert!(super::matches_right_context("CBA", 'A' as u8, &vec!['B' as u8, 'C' as u8]));
    }

    #[test]
    fn matches_right_context_test_branch() {
        assert!(super::matches_right_context("[B]A", 'A' as u8, &vec![]));
        assert!(super::matches_right_context("[[B]C][D[E]]A", 'A' as u8, &vec![]));
        assert!(!super::matches_right_context("[A]", 'A' as u8, &vec![]));
        assert!(!super::matches_right_context("]A", 'A' as u8, &vec![]));
        assert!(!super::matches_right_context("[BC]]A", 'A' as u8, &vec![]));
    }

    #[test]
    fn matches_right_context_test_branch_ignore() {
        assert!(super::matches_right_context("F[[B]C]F[D[E]]FA", 'A' as u8, &vec!['C' as u8, 'F' as u8]));
    }

    #[test]
    fn matches_predecessor_test_full() {
        let pred = super::Predecessor {
            strict: 'A' as u8,
            left: Some('B' as u8),
            right: Some('C' as u8),
        };
        assert!(!super::matches_predecessor("BAC", 0, &pred, &vec![]));
        assert!(super::matches_predecessor("BAC", 1, &pred, &vec![]));
        assert!(!super::matches_predecessor("BAC", 2, &pred, &vec![]));
    }

    #[test]
    fn matches_predecessor_test_left() {
        let pred = super::Predecessor {
            strict: 'A' as u8,
            left: Some('B' as u8),
            right: None,
        };
        assert!(!super::matches_predecessor("BAD", 0, &pred, &vec![]));
        assert!(super::matches_predecessor("BAD", 1, &pred, &vec![]));
        assert!(!super::matches_predecessor("BAD", 2, &pred, &vec![]));
    }

    #[test]
    fn matches_predecessor_test_right() {
        let pred = super::Predecessor {
            strict: 'A' as u8,
            left: None,
            right: Some('C' as u8),
        };
        assert!(!super::matches_predecessor("DAC", 0, &pred, &vec![]));
        assert!(super::matches_predecessor("DAC", 1, &pred, &vec![]));
        assert!(!super::matches_predecessor("DAC", 2, &pred, &vec![]));
    }

    #[test]
    fn matches_predecessor_test_none() {
        let pred = super::Predecessor {
            strict: 'A' as u8,
            left: None,
            right: None,
        };
        assert!(!super::matches_predecessor("BAC", 0, &pred, &vec![]));
        assert!(super::matches_predecessor("BAC", 1, &pred, &vec![]));
        assert!(!super::matches_predecessor("BAC", 2, &pred, &vec![]));
    }
}
