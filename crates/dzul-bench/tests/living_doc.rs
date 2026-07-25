//! Integration tests for verifying living documentation anchors and requirements.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Verifies that all requirements in `requirements.md` have corresponding code anchors and tests.
#[test]
fn test_living_doc() {
    let req_path = Path::new("spec/requirements.md");
    if !req_path.exists() {
        println!("requirements.md not found at spec/requirements.md");
        return;
    }
    let req_content = fs::read_to_string(req_path).unwrap();
    let mut expected_ids = HashSet::new();
    for line in req_content.lines() {
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if c == 'F' && chars.peek() == Some(&'R') {
                chars.next();
                if chars.peek() == Some(&'-') {
                    chars.next();
                    let mut num = String::new();
                    while let Some(&digit) = chars.peek() {
                        if digit.is_ascii_digit() {
                            num.push(digit);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if !num.is_empty() {
                        expected_ids.insert(format!("FR-{num}"));
                    }
                }
            } else if c == 'N' && chars.peek() == Some(&'F') {
                chars.next();
                if chars.peek() == Some(&'R') {
                    chars.next();
                    if chars.peek() == Some(&'-') {
                        chars.next();
                        let mut num = String::new();
                        while let Some(&digit) = chars.peek() {
                            if digit.is_ascii_digit() {
                                num.push(digit);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        if !num.is_empty() {
                            expected_ids.insert(format!("NFR-{num}"));
                        }
                    }
                }
            }
        }
    }

    assert!(
        !expected_ids.is_empty(),
        "No requirements found in requirements.md"
    );

    let mut found_anchors = HashSet::new();
    let mut found_tests = HashSet::new();

    scan_dir(Path::new("crates/dzul-core/src"), &mut found_anchors, &mut found_tests);
    scan_dir(Path::new("crates/dzul-bench"), &mut found_anchors, &mut found_tests);

    let mut missing = Vec::new();
    for id in &expected_ids {
        let has_anchor = found_anchors.contains(id);
        let has_test = found_tests.contains(id);
        if !has_anchor || !has_test {
            missing.push(format!("ID: {id} (Anchor: {has_anchor}, Test: {has_test})"));
        }
    }

    assert!(
        missing.is_empty(),
        "Missing living documentation requirements:\n{}",
        missing.join("\n")
    );
}

fn scan_dir(
    dir: &Path,
    found_anchors: &mut HashSet<String>,
    found_tests: &mut HashSet<String>,
) {
    if !dir.exists() {
        return;
    }
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, found_anchors, found_tests);
        } else if path.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            let content = fs::read_to_string(&path).unwrap();
            for line in content.lines() {
                if let Some(idx) = line.find("Anchor:") {
                    let anchor = line[idx + 7..]
                        .trim()
                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
                    if !anchor.is_empty() {
                        found_anchors.insert(anchor.to_string());
                    }
                }
                if line.contains("fn test_") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    for part in parts {
                        if part.starts_with("test_") {
                            let test_name = part.trim_end_matches('(').trim();
                            let normalized = test_name.replace('_', "-").to_uppercase();
                            if let Some(idx) = normalized.find("FR-") {
                                if idx >= 1
                                    && &normalized[idx - 1..idx] == "N"
                                    && normalized.len() >= idx + 5
                                {
                                    found_tests
                                        .insert(normalized[idx - 1..idx + 5].to_string());
                                } else if normalized.len() >= idx + 5 {
                                    found_tests.insert(normalized[idx..idx + 5].to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
