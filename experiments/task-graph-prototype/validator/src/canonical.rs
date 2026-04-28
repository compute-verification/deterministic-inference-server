//! Canonical-JSON encoder matching Python's
//! `json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + "\n"`.
//!
//! Keys are sorted lexicographically by Unicode code point. Strings escape
//! control characters and the standard JSON specials (`"` and `\`).
//! For the prototype all string values are ASCII, so the
//! `ensure_ascii=True` high-codepoint branch is included but exercised only
//! by future inputs.

use serde_json::Value;

pub fn canonical_bytes(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_value(&mut out, v);
    out.push(b'\n');
    out
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::String(s) => write_string(out, s),
        Value::Array(arr) => {
            out.push(b'[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, item);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_string(out, k);
                out.push(b':');
                write_value(out, &map[*k]);
            }
            out.push(b'}');
        }
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\x08' => out.extend_from_slice(b"\\b"),
            '\x09' => out.extend_from_slice(b"\\t"),
            '\x0a' => out.extend_from_slice(b"\\n"),
            '\x0c' => out.extend_from_slice(b"\\f"),
            '\x0d' => out.extend_from_slice(b"\\r"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c if (c as u32) > 0x7e => {
                // ensure_ascii=True branch: escape every non-ASCII code point.
                let mut buf = [0u16; 2];
                let units = c.encode_utf16(&mut buf);
                for u in units.iter() {
                    out.extend_from_slice(format!("\\u{:04x}", *u).as_bytes());
                }
            }
            c => {
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                out.extend_from_slice(bytes);
            }
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_simple_object() {
        // Python:
        //   json.dumps({"b":1,"a":2}, sort_keys=True, separators=(",", ":"),
        //              ensure_ascii=True) + "\n"
        // -> '{"a":2,"b":1}\n'
        let v = json!({"b": 1, "a": 2});
        assert_eq!(canonical_bytes(&v), b"{\"a\":2,\"b\":1}\n");
    }

    #[test]
    fn nested_sort() {
        let v = json!({"x":[{"b":1,"a":2}],"y":null});
        assert_eq!(
            canonical_bytes(&v),
            b"{\"x\":[{\"a\":2,\"b\":1}],\"y\":null}\n"
        );
    }

    #[test]
    fn escapes_quotes_backslashes() {
        let v = json!({"k": "he said \"hi\\there\""});
        assert_eq!(
            canonical_bytes(&v),
            "{\"k\":\"he said \\\"hi\\\\there\\\"\"}\n".as_bytes()
        );
    }
}
