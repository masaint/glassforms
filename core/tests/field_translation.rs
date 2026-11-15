//! The WGSL of every field preset, checked against the GLSL the web version compiles, by running both.
//!
//! No GPU is involved. A small interpreter (below) reads the part of GLSL and the part of WGSL that these
//! shaders are written in, and works them out in 32-bit floats, operation by operation in the order
//! written. The GLSL is the very text the web version hands to WebGL (`data/fields.json`, written by
//! `tools/dump_fields.mjs`); the WGSL is what `field::bake_shader` makes. Both are run on the same
//! points with the same numbers and must give the same bits: the two use the same sin, cos and pow here,
//! so any difference at all is a difference in the code.
//!
//! What this does not show: that a GPU's own sin, cos or rounding agree between the two APIs.
use std::collections::{BTreeMap, HashMap};

use gf_core::field;
use gf_core::presets::{self, Family, Preset, Shape};
use serde_json::Value as Json;

// ------------------------------------------------------------------ reading shader code
#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Id(String),
    Num(String),
    P(&'static str),
}

const PUNCT: &[&str] = &[
    "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "++", "--", "->",
    "+", "-", "*", "/", "%", "<", ">", "=", "!", "&", "|", "?", ":", ";", ",", ".", "(", ")", "{", "}", "[", "]", "@",
];

fn lex(src: &str) -> Vec<Tok> {
    let s: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && s.get(i + 1) == Some(&'/') {
            while i < s.len() && s[i] != '\n' { i += 1; }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let a = i;
            while i < s.len() && (s[i].is_ascii_alphanumeric() || s[i] == '_') { i += 1; }
            out.push(Tok::Id(s[a..i].iter().collect()));
        } else if c.is_ascii_digit() {
            let a = i;
            while i < s.len() && s[i].is_ascii_digit() { i += 1; }
            if i < s.len() && s[i] == '.' {
                i += 1;
                while i < s.len() && s[i].is_ascii_digit() { i += 1; }
            }
            if i < s.len() && (s[i] == 'e' || s[i] == 'E') {
                let j = if matches!(s.get(i + 1), Some('+') | Some('-')) { i + 2 } else { i + 1 };
                if s.get(j).is_some_and(|c| c.is_ascii_digit()) {
                    i = j;
                    while i < s.len() && s[i].is_ascii_digit() { i += 1; }
                }
            }
            if i < s.len() && matches!(s[i], 'u' | 'i' | 'f') { i += 1; }       // 1u, 2.0f
            out.push(Tok::Num(s[a..i].iter().collect()));
        } else {
            let next: String = s[i..(i + 2).min(s.len())].iter().collect();
            let p = PUNCT.iter().find(|p| next.starts_with(**p)).unwrap_or_else(|| panic!("cannot read {next:?}"));
            out.push(Tok::P(p));
            i += p.len();
        }
    }
    out
}

#[derive(Clone, Debug)]
enum Ex {
    Num(String),
    Id(String),
    Member(Box<Ex>, String),
    Index(Box<Ex>, Box<Ex>),
    Neg(Box<Ex>),
    Bin(&'static str, Box<Ex>, Box<Ex>),
    Cond(Box<Ex>, Box<Ex>, Box<Ex>),
    Call(String, Vec<Ex>),
}

#[derive(Clone, Debug)]
enum St {
    /// A new name. `untyped_const`: WGSL's `const x = ...` with no type, which stays a literal number.
    Decl { name: String, init: Option<Ex>, untyped_const: bool },
    /// target, "=" or "+=" and the like, value.  `i++` is read as `i += 1`.
    Assign(Ex, &'static str, Ex),
    If(Ex, Vec<St>, Vec<St>),
    /// start, condition, step, body
    For(Vec<St>, Ex, Vec<St>, Vec<St>),
    Break,
    Return(Option<Ex>),
}

struct Func {
    params: Vec<String>,
    body: Vec<St>,
}

#[derive(Default)]
struct Program {
    wgsl: bool,
    funcs: HashMap<String, Func>,
    /// GLSL: `#define name text`. The text is worked out wherever the name is used, with what is in reach there.
    macros: HashMap<String, Ex>,
    /// WGSL: each struct's members in order, as (name, type)
    structs: HashMap<String, Vec<(String, String)>>,
    /// variables declared outside any function, as (name, type)
    globals: Vec<(String, String)>,
}

// binary operators, loosest first; all of them group from the left
const LEVELS: &[&[&str]] = &[&["&"], &["<", ">", "<=", ">=", "==", "!="], &["<<", ">>"], &["+", "-"], &["*", "/", "%"]];
const GLSL_TYPES: &[&str] = &["float", "int", "bool", "vec2", "vec3", "vec4"];

fn bx(e: Ex) -> Box<Ex> { Box::new(e) }

struct Parser {
    t: Vec<Tok>,
    i: usize,
    wgsl: bool,
}
impl Parser {
    fn peek(&self) -> Option<&Tok> { self.t.get(self.i) }
    fn is(&self, p: &str) -> bool { matches!(self.peek(), Some(Tok::P(q)) if *q == p) }
    fn eat(&mut self, p: &str) -> bool {
        let yes = self.is(p);
        if yes { self.i += 1; }
        yes
    }
    fn eat_id(&mut self, word: &str) -> bool {
        let yes = matches!(self.peek(), Some(Tok::Id(q)) if q == word);
        if yes { self.i += 1; }
        yes
    }
    fn stuck(&self, what: &str) -> ! {
        let from = self.i.saturating_sub(6);
        panic!("expected {what} after {:?}, found {:?}", &self.t[from..self.i], self.peek())
    }
    fn expect(&mut self, p: &str) {
        if !self.eat(p) { self.stuck(p); }
    }
    fn ident(&mut self) -> String {
        match self.peek() {
            Some(Tok::Id(s)) => { let s = s.clone(); self.i += 1; s }
            _ => self.stuck("a name"),
        }
    }
    /// skip from just inside an opening bracket to just past the one that closes it
    fn skip_to_close(&mut self, open: &str, close: &str) {
        let mut depth = 1;
        while depth > 0 {
            if self.eat(open) { depth += 1; } else if self.eat(close) { depth -= 1; } else { self.i += 1; }
            if self.i > self.t.len() { self.stuck(close); }
        }
    }
    // WGSL: @vertex, @builtin(position), @location(0) ...
    fn attributes(&mut self) {
        while self.eat("@") {
            self.ident();
            if self.eat("(") { self.skip_to_close("(", ")"); }
        }
    }
    // WGSL: f32, vec3f, vec3<f32>, a struct's name
    fn ty(&mut self) -> String {
        let name = self.ident();
        if self.eat("<") { self.skip_to_close("<", ">"); }
        name
    }

    fn expr(&mut self) -> Ex {
        let c = self.binary(0);
        if self.eat("?") {
            let a = self.expr();
            self.expect(":");
            let b = self.expr();
            return Ex::Cond(bx(c), bx(a), bx(b));
        }
        c
    }
    fn binary(&mut self, level: usize) -> Ex {
        if level == LEVELS.len() { return self.unary(); }
        let mut left = self.binary(level + 1);
        loop {
            let op: &'static str = match self.peek() {
                Some(Tok::P(p)) if LEVELS[level].contains(p) => p,
                _ => return left,
            };
            self.i += 1;
            let right = self.binary(level + 1);
            left = Ex::Bin(op, bx(left), bx(right));
        }
    }
    fn unary(&mut self) -> Ex {
        if self.eat("-") { return Ex::Neg(bx(self.unary())); }
        self.postfix()
    }
    fn postfix(&mut self) -> Ex {
        let mut e = match self.peek().cloned() {
            Some(Tok::Num(n)) => { self.i += 1; Ex::Num(n) }
            Some(Tok::Id(name)) => {
                self.i += 1;
                if self.eat("(") {
                    let mut args = Vec::new();
                    while !self.eat(")") {
                        args.push(self.expr());
                        if !self.is(")") { self.expect(","); }
                    }
                    Ex::Call(name, args)
                } else {
                    Ex::Id(name)
                }
            }
            Some(Tok::P("(")) => {
                self.i += 1;
                let e = self.expr();
                self.expect(")");
                e
            }
            _ => self.stuck("a value"),
        };
        loop {
            if self.eat(".") {
                e = Ex::Member(bx(e), self.ident());
            } else if self.eat("[") {
                let i = self.expr();
                self.expect("]");
                e = Ex::Index(bx(e), bx(i));
            } else {
                return e;
            }
        }
    }

    /// `{ ... }`, or one statement standing alone
    fn body(&mut self) -> Vec<St> {
        let mut out = Vec::new();
        if self.eat("{") {
            while !self.eat("}") { self.stmt(&mut out); }
        } else {
            self.stmt(&mut out);
        }
        out
    }
    fn stmt(&mut self, out: &mut Vec<St>) {
        if self.eat_id("return") {
            let e = if self.is(";") { None } else { Some(self.expr()) };
            self.expect(";");
            out.push(St::Return(e));
        } else if self.eat_id("break") {
            self.expect(";");
            out.push(St::Break);
        } else if self.eat_id("if") {
            let c = if self.wgsl {
                self.expr()
            } else {
                self.expect("(");
                let c = self.expr();
                self.expect(")");
                c
            };
            let then = self.body();
            let other = if self.eat_id("else") { self.body() } else { Vec::new() };
            out.push(St::If(c, then, other));
        } else if self.eat_id("for") {
            self.expect("(");
            let mut start = Vec::new();
            self.simple(&mut start);
            self.expect(";");
            let cond = self.expr();
            self.expect(";");
            let mut step = Vec::new();
            self.simple(&mut step);
            self.expect(")");
            let body = self.body();
            out.push(St::For(start, cond, step, body));
        } else {
            self.simple(out);
            self.expect(";");
        }
    }
    /// a declaration or an assignment, without the semicolon
    fn simple(&mut self, out: &mut Vec<St>) {
        if self.wgsl {
            for word in ["let", "var", "const"] {
                if self.eat_id(word) {
                    let name = self.ident();
                    let typed = self.eat(":");
                    if typed { self.ty(); }
                    let init = if self.eat("=") { Some(self.expr()) } else { None };
                    out.push(St::Decl { name, init, untyped_const: word == "const" && !typed });
                    return;
                }
            }
        } else {
            self.eat_id("const");
            // "vec3 a = ..., b = ...": a type and then a name (a type and then a bracket is a value being made)
            let a_type = matches!(self.peek(), Some(Tok::Id(t)) if GLSL_TYPES.contains(&t.as_str()));
            if a_type && matches!(self.t.get(self.i + 1), Some(Tok::Id(_))) {
                self.i += 1;
                loop {
                    let name = self.ident();
                    let init = if self.eat("=") { Some(self.expr()) } else { None };
                    out.push(St::Decl { name, init, untyped_const: false });
                    if !self.eat(",") { return; }
                }
            }
        }
        let target = self.postfix();
        if self.eat("++") {
            out.push(St::Assign(target, "+=", Ex::Num("1".to_string())));
            return;
        }
        let Some(op) = ["=", "+=", "-=", "*=", "/="].into_iter().find(|op| self.eat(op)) else { self.stuck("an assignment") };
        let value = self.expr();
        out.push(St::Assign(target, op, value));
    }
}

/// One expression on its own.
fn parse_expr(src: &str, wgsl: bool) -> Ex {
    let mut p = Parser { t: lex(src), i: 0, wgsl };
    let e = p.expr();
    assert!(p.peek().is_none(), "left over after the expression {src:?}: {:?}", p.peek());
    e
}

/// A whole GLSL ES 3.0 shader, of the kind forms.js writes.
fn parse_glsl(src: &str) -> Program {
    let mut prog = Program::default();
    let mut code = String::new();
    for line in src.lines() {
        let line = line.split("//").next().unwrap();
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#define ") {
            let (name, body) = rest.trim().split_once(char::is_whitespace).unwrap_or_else(|| panic!("a #define with nothing in it: {t}"));
            prog.macros.insert(name.to_string(), parse_expr(body, false));
        } else if !t.starts_with('#') {
            code.push_str(line);
            code.push('\n');
        }
    }
    let mut p = Parser { t: lex(&code), i: 0, wgsl: false };
    while p.peek().is_some() {
        if p.eat_id("precision") || p.eat_id("uniform") {
            while !p.eat(";") { p.i += 1; }             // uniforms are given their values by whoever runs the shader
        } else if p.eat_id("out") {
            let ty = p.ident();
            let name = p.ident();
            p.expect(";");
            prog.globals.push((name, ty));
        } else {
            let _returns = p.ident();
            let name = p.ident();
            p.expect("(");
            let mut params = Vec::new();
            while !p.eat(")") {
                let _ty = p.ident();
                params.push(p.ident());
                if !p.is(")") { p.expect(","); }
            }
            assert!(p.is("{"), "{name}: a function needs a body");
            let body = p.body();
            prog.funcs.insert(name, Func { params, body });
        }
    }
    prog
}

/// A whole WGSL module, of the kind `field::bake_shader` writes.
fn parse_wgsl(src: &str) -> Program {
    let mut prog = Program { wgsl: true, ..Default::default() };
    let mut p = Parser { t: lex(src), i: 0, wgsl: true };
    while p.peek().is_some() {
        p.attributes();
        if p.eat_id("struct") {
            let name = p.ident();
            p.expect("{");
            let mut members = Vec::new();
            while !p.eat("}") {
                p.attributes();
                let member = p.ident();
                p.expect(":");
                members.push((member, p.ty()));
                p.eat(",");
            }
            p.eat(";");
            prog.structs.insert(name, members);
        } else if p.eat_id("var") {
            if p.eat("<") { p.skip_to_close("<", ">"); }
            let name = p.ident();
            p.expect(":");
            let ty = p.ty();
            p.expect(";");
            prog.globals.push((name, ty));
        } else if p.eat_id("fn") {
            let name = p.ident();
            p.expect("(");
            let mut params = Vec::new();
            while !p.eat(")") {
                p.attributes();
                params.push(p.ident());
                p.expect(":");
                p.ty();
                if !p.is(")") { p.expect(","); }
            }
            if p.eat("->") {
                p.attributes();
                p.ty();
            }
            assert!(p.is("{"), "{name}: a function needs a body");
            let body = p.body();
            prog.funcs.insert(name, Func { params, body });
        } else {
            p.stuck("struct, var or fn");
        }
    }
    prog
}

// ------------------------------------------------------------------ running it
#[derive(Clone, Debug, PartialEq)]
enum Val {
    /// a float as the GPU holds it
    F(f32),
    /// WGSL only: a number written out in the code (or sums and products of such) that has no type yet.
    /// It is worked out in 64 bits and becomes an f32 when it meets one, as the language says.
    A(f64),
    I(i64),
    B(bool),
    /// vec2, vec3, vec4; also GLSL's `float uPr[8]`
    V(Vec<f32>),
    /// a WGSL struct
    S(BTreeMap<String, Val>),
}
use Val::{A, B, F, I, S, V};

fn concrete(v: Val) -> Val {
    match v {
        A(x) => F(x as f32),
        v => v,
    }
}

fn number(text: &str, wgsl: bool) -> Val {
    let (digits, suffix) = match text.chars().last() {
        Some(c @ ('u' | 'i' | 'f')) => (&text[..text.len() - 1], Some(c)),
        _ => (text, None),
    };
    let float = suffix == Some('f') || digits.contains(['.', 'e', 'E']);
    if !float { return I(digits.parse().unwrap()); }
    if wgsl && suffix.is_none() { A(digits.parse().unwrap()) } else { F(digits.parse().unwrap()) }
}

fn arith(op: &str, x: f32, y: f32) -> f32 {
    match op {
        "+" => x + y,
        "-" => x - y,
        "*" => x * y,
        "/" => x / y,
        _ => panic!("no {op} between floats"),
    }
}
fn compare(op: &str, x: f64, y: f64) -> bool {
    match op {
        "<" => x < y,
        ">" => x > y,
        "<=" => x <= y,
        ">=" => x >= y,
        "==" => x == y,
        "!=" => x != y,
        _ => panic!("no {op} between numbers"),
    }
}
fn binary(op: &str, a: Val, b: Val) -> Val {
    let sums = matches!(op, "+" | "-" | "*" | "/");
    match (a, b) {
        (A(x), A(y)) => match op {
            "+" => A(x + y),
            "-" => A(x - y),
            "*" => A(x * y),
            "/" => A(x / y),
            _ => B(compare(op, x, y)),
        },
        (I(x), I(y)) => match op {
            "+" => I(x + y),
            "-" => I(x - y),
            "*" => I(x * y),
            "<<" => I(x << y),
            "&" => I(x & y),
            _ => B(compare(op, x as f64, y as f64)),
        },
        (a, b) => match (concrete(a), concrete(b)) {
            (F(x), F(y)) => if sums { F(arith(op, x, y)) } else { B(compare(op, x as f64, y as f64)) },
            (V(x), F(y)) => V(x.iter().map(|x| arith(op, *x, y)).collect()),
            (F(x), V(y)) => V(y.iter().map(|y| arith(op, x, *y)).collect()),
            (V(x), V(y)) if x.len() == y.len() => V(x.iter().zip(&y).map(|(x, y)| arith(op, *x, *y)).collect()),
            (a, b) => panic!("cannot work out {a:?} {op} {b:?}"),
        },
    }
}

fn member(v: Val, m: &str) -> Val {
    match v {
        S(s) => s.get(m).cloned().unwrap_or_else(|| panic!("no member called {m}")),
        V(v) => {
            let pick: Vec<f32> = m.chars().map(|c| v[match c {
                'x' | 'r' => 0,
                'y' | 'g' => 1,
                'z' | 'b' => 2,
                'w' | 'a' => 3,
                _ => panic!("no component called {c}"),
            }]).collect();
            if pick.len() == 1 { F(pick[0]) } else { V(pick) }
        }
        other => panic!("{other:?} has no .{m}"),
    }
}

fn vector(n: usize, parts: &[Val]) -> Val {
    let mut out = Vec::new();
    for v in parts {
        match v {
            F(x) => out.push(*x),
            V(v) => out.extend(v),
            other => panic!("a vector cannot hold {other:?}"),
        }
    }
    if out.len() == 1 { out = vec![out[0]; n]; }
    assert_eq!(out.len(), n, "wrong number of values for a vec{n}");
    V(out)
}

#[allow(clippy::manual_clamp)]      // clamp and smoothstep are spelled out as the languages define them
/// The functions the language provides. Each is worked out place by place; a lone number stands for
/// every place of a vector (GLSL allows that in min, max, mix and clamp; in WGSL the compiler, not
/// this, is what insists on matching types).
fn builtin(name: &str, args: Vec<Val>) -> Val {
    let makes = matches!(name, "vec2" | "vec3" | "vec4" | "vec2f" | "vec3f" | "vec4f" | "float" | "f32");
    assert!(makes || !args.iter().all(|a| matches!(a, A(_))), "{name} of nothing but written-out numbers: not something this interpreter does");
    let a: Vec<Val> = args.into_iter().map(concrete).collect();
    let n = a.iter().map(|v| if let V(v) = v { v.len() } else { 1 }).max().unwrap_or(1);
    let at = |k: usize, i: usize| -> f32 {
        match &a[k] {
            F(x) => *x,
            V(v) => v[i],
            other => panic!("{name}: {other:?} is not a number"),
        }
    };
    let each = |f: &dyn Fn(usize) -> f32| -> Val { if n == 1 { F(f(0)) } else { V((0..n).map(f).collect()) } };
    match (name, a.len()) {
        ("sin", 1) => each(&|i| at(0, i).sin()),
        ("cos", 1) => each(&|i| at(0, i).cos()),
        ("abs", 1) => each(&|i| at(0, i).abs()),
        ("min", 2) => each(&|i| at(0, i).min(at(1, i))),
        ("max", 2) => each(&|i| at(0, i).max(at(1, i))),
        ("pow", 2) => each(&|i| at(0, i).powf(at(1, i))),
        ("clamp", 3) => each(&|i| at(0, i).max(at(1, i)).min(at(2, i))),
        ("mix", 3) => each(&|i| at(0, i) * (1.0 - at(2, i)) + at(1, i) * at(2, i)),
        ("smoothstep", 3) => each(&|i| {
            let t = ((at(2, i) - at(0, i)) / (at(1, i) - at(0, i))).max(0.0).min(1.0);
            t * t * (3.0 - 2.0 * t)
        }),
        ("length", 1) => F((0..n).map(|i| at(0, i) * at(0, i)).sum::<f32>().sqrt()),
        ("dot", 2) => F((0..n).map(|i| at(0, i) * at(1, i)).sum::<f32>()),
        ("select", 3) => match &a[2] {
            B(yes) => a[if *yes { 1 } else { 0 }].clone(),
            other => panic!("select on {other:?}"),
        },
        ("float" | "f32", 1) => match &a[0] {
            I(i) => F(*i as f32),
            F(x) => F(*x),
            other => panic!("{name} of {other:?}"),
        },
        ("vec2" | "vec2f", _) => vector(2, &a),
        ("vec3" | "vec3f", _) => vector(3, &a),
        ("vec4" | "vec4f", _) => vector(4, &a),
        _ => panic!("no function {name} taking {} values", a.len()),
    }
}

enum Flow {
    Next,
    Break,
    Ret(Val),
}

struct Machine<'a> {
    prog: &'a Program,
    globals: HashMap<String, Val>,
    /// the names in reach inside the function being run, innermost block last
    scopes: Vec<HashMap<String, Val>>,
}
impl<'a> Machine<'a> {
    fn new(prog: &'a Program) -> Self {
        let globals = prog.globals.iter().map(|(name, ty)| {
            let zero = match ty.as_str() {
                "vec4" | "vec4f" => V(vec![0.0; 4]),
                "float" | "f32" => F(0.0),
                _ => S(BTreeMap::new()),
            };
            (name.clone(), zero)
        }).collect();
        Machine { prog, globals, scopes: Vec::new() }
    }
    fn set(&mut self, name: &str, v: Val) { self.globals.insert(name.to_string(), v); }

    fn lookup(&mut self, name: &str) -> Val {
        let prog = self.prog;
        if let Some(text) = prog.macros.get(name) { return self.eval(text); }
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(name) { return v.clone(); }
        }
        self.globals.get(name).cloned().unwrap_or_else(|| panic!("nothing is called {name}"))
    }
    fn slot(&mut self, name: &str) -> &mut Val {
        for s in self.scopes.iter_mut().rev() {
            if let Some(v) = s.get_mut(name) { return v; }
        }
        self.globals.get_mut(name).unwrap_or_else(|| panic!("nothing to assign to is called {name}"))
    }
    fn store(&mut self, target: &Ex, v: Val) {
        match target {
            Ex::Id(name) => *self.slot(name) = v,
            Ex::Member(of, m) => match (&**of, m) {
                (Ex::Id(name), m) => match self.slot(name) {
                    S(s) => { s.insert(m.clone(), v); }
                    other => panic!("cannot set .{m} of {other:?}"),
                },
                _ => panic!("cannot assign to {target:?}"),
            },
            _ => panic!("cannot assign to {target:?}"),
        }
    }

    fn eval(&mut self, e: &Ex) -> Val {
        match e {
            Ex::Num(text) => number(text, self.prog.wgsl),
            Ex::Id(name) => self.lookup(name),
            Ex::Member(of, m) => member(self.eval(of), m),
            Ex::Index(of, i) => match (self.eval(of), self.eval(i)) {
                (V(v), I(i)) => F(v[i as usize]),
                (v, i) => panic!("cannot take [{i:?}] of {v:?}"),
            },
            Ex::Neg(of) => match self.eval(of) {
                F(x) => F(-x),
                A(x) => A(-x),
                I(x) => I(-x),
                V(v) => V(v.iter().map(|x| -x).collect()),
                other => panic!("cannot negate {other:?}"),
            },
            Ex::Bin(op, a, b) => {
                let a = self.eval(a);
                let b = self.eval(b);
                binary(op, a, b)
            }
            Ex::Cond(c, a, b) => match self.eval(c) {
                B(true) => self.eval(a),
                B(false) => self.eval(b),
                other => panic!("a condition that is {other:?}"),
            },
            Ex::Call(name, args) => {
                let args = args.iter().map(|a| self.eval(a)).collect();
                self.call(name, args)
            }
        }
    }
    fn call(&mut self, name: &str, args: Vec<Val>) -> Val {
        let prog = self.prog;
        let Some(f) = prog.funcs.get(name) else { return builtin(name, args) };
        assert_eq!(f.params.len(), args.len(), "{name}: wrong number of values");
        let frame = f.params.iter().cloned().zip(args.into_iter().map(concrete)).collect();
        let outer = std::mem::replace(&mut self.scopes, vec![frame]);
        let result = match self.run(&f.body) {
            Flow::Ret(v) => v,
            _ => B(false),              // a function that returns nothing
        };
        self.scopes = outer;
        result
    }
    fn block(&mut self, body: &[St]) -> Flow {
        self.scopes.push(HashMap::new());
        let flow = self.run(body);
        self.scopes.pop();
        flow
    }
    fn run(&mut self, body: &[St]) -> Flow {
        for s in body {
            match s {
                St::Decl { name, init, untyped_const } => {
                    let v = match init {
                        Some(e) => {
                            let v = self.eval(e);
                            if *untyped_const { v } else { concrete(v) }
                        }
                        None => S(BTreeMap::new()),         // WGSL's `var o: VOut;`, filled in member by member
                    };
                    self.scopes.last_mut().unwrap().insert(name.clone(), v);
                }
                St::Assign(target, op, e) => {
                    let mut v = self.eval(e);
                    if *op != "=" {
                        let now = self.eval(target);
                        v = binary(&op[..1], now, v);
                    }
                    self.store(target, concrete(v));
                }
                St::If(c, then, other) => {
                    let branch = match self.eval(c) {
                        B(true) => then,
                        B(false) => other,
                        v => panic!("a condition that is {v:?}"),
                    };
                    match self.block(branch) {
                        Flow::Next => {}
                        flow => return flow,
                    }
                }
                St::For(start, cond, step, body) => {
                    self.scopes.push(HashMap::new());
                    self.run(start);
                    let mut returned = None;
                    while self.eval(cond) == B(true) {
                        match self.block(body) {
                            Flow::Next => {}
                            Flow::Break => break,
                            Flow::Ret(v) => { returned = Some(v); break; }
                        }
                        self.run(step);
                    }
                    self.scopes.pop();
                    if let Some(v) = returned { return Flow::Ret(v); }
                }
                St::Break => return Flow::Break,
                St::Return(e) => return Flow::Ret(match e {
                    Some(e) => concrete(self.eval(e)),
                    None => B(false),
                }),
            }
        }
        Flow::Next
    }
    /// An expression in which `p` is this point.
    fn at_point(&mut self, e: &Ex, p: [f32; 3]) -> Val {
        self.scopes = vec![HashMap::from([("p".to_string(), V(p.to_vec()))])];
        let v = concrete(self.eval(e));
        self.scopes.clear();
        v
    }
}

// ------------------------------------------------------------------ the two shaders of one preset
/// What a bake is told: hb, res, puff, a spare, then eight parameters (as `field::uniforms` lays them out).
type Uniforms = [f32; 12];

/// The web version's shader, given its uniforms by name.
fn glsl_machine<'a>(prog: &'a Program, u: &Uniforms) -> Machine<'a> {
    let mut m = Machine::new(prog);
    m.set("uHB", F(u[0]));
    m.set("uRes", F(u[1]));
    m.set("uPuff", F(u[2]));
    m.set("uPr", V(u[4..].to_vec()));
    m
}

/// The native shader, given its uniform block: the twelve numbers poured, in order, into the structs the
/// shader itself declares.
fn wgsl_machine<'a>(prog: &'a Program, u: &Uniforms) -> Machine<'a> {
    fn pour(prog: &Program, ty: &str, from: &mut dyn Iterator<Item = f32>) -> Val {
        match prog.structs.get(ty) {
            Some(members) => S(members.iter().map(|(name, ty)| (name.clone(), pour(prog, ty, from))).collect()),
            None => {
                assert_eq!(ty, "f32", "the uniform block should hold nothing but floats");
                F(from.next().expect("the uniform block is longer than the numbers given"))
            }
        }
    }
    let mut m = Machine::new(prog);
    let ty = &prog.globals.iter().find(|(name, _)| name == "u").expect("no uniform block called u").1;
    let mut numbers = u.iter().copied();
    let block = pour(prog, ty, &mut numbers);
    assert!(numbers.next().is_none(), "the uniform block is shorter than the numbers given");
    m.set("u", block);
    m
}

struct Pair<'a> {
    family: &'a str,
    preset: &'a Preset,
    glsl: Program,
    wgsl: Program,
    clip: Ex,
    shell: Ex,
}

fn dump() -> Json {
    serde_json::from_str(include_str!("data/fields.json")).unwrap()
}
fn families() -> Vec<Family> {
    vec![presets::surfaces::family(), presets::algebraic::family()]
}
fn pairs<'a>(fams: &'a [Family], dump: &Json) -> Vec<Pair<'a>> {
    let mut out = Vec::new();
    for fam in fams {
        let dumped = dump["families"].as_array().unwrap().iter().find(|f| f["id"] == fam.id).unwrap();
        for p in &fam.presets {
            let d = dumped["presets"].as_array().unwrap().iter().find(|d| d["id"] == p.id).unwrap_or_else(|| panic!("{} is not in fields.json", p.id));
            let Shape::Field { clip, shell, .. } = &p.shape else { panic!("{}: not an equation", p.id) };
            out.push(Pair {
                family: fam.id,
                preset: p,
                glsl: parse_glsl(d["fragment"].as_str().unwrap()),
                wgsl: parse_wgsl(&field::bake_shader(p)),
                clip: parse_expr(clip, true),
                shell: parse_expr(shell, true),
            });
        }
    }
    out
}

/// Numbers that look random and are the same on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn between(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.next() }
    fn below(&mut self, n: u32) -> u32 { (self.next() * n as f64) as u32 }
}

/// Four settings of a preset's sliders, and an inflation to go with each. The first is the starting
/// values. In the others every smooth slider is somewhere in its range, and the switches (sliders that
/// only have two ends, like solid, cube and side) are thrown in every combination.
fn settings(p: &Preset, rng: &mut Rng) -> Vec<(Vec<f64>, f64)> {
    let puffs = [0.0, 0.02, -0.03, 0.05];
    (0..4).map(|s| {
        let mut switch = 0;
        let values = p.params.iter().map(|d| {
            if d.step >= 1.0 {
                let thrown = (s >> switch) & 1 == 1;
                switch += 1;
                if thrown { if d.value == d.lo { d.hi } else { d.lo } } else { d.value }
            } else if s == 0 {
                d.value
            } else {
                rng.between(d.lo, d.hi)
            }
        }).collect();
        (values, puffs[s])
    }).collect()
}

/// A tally of comparisons: how many, how many differed, and the largest difference.
#[derive(Default)]
struct Tally {
    n: usize,
    bad: usize,
    worst: f64,
    first: Option<String>,
}
impl Tally {
    fn check(&mut self, what: impl Fn() -> String, glsl: &Val, wgsl: &Val) {
        let flat = |v: &Val| -> Vec<f32> {
            match v {
                F(x) => vec![*x],
                V(v) => v.clone(),
                other => panic!("{}: got {other:?}", what()),
            }
        };
        let (a, b) = (flat(glsl), flat(wgsl));
        assert_eq!(a.len(), b.len(), "{}", what());
        for (x, y) in a.iter().zip(&b) {
            self.n += 1;
            if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
                self.bad += 1;
                let d = (*x as f64 - *y as f64).abs();
                self.worst = if d.is_nan() { f64::INFINITY } else { self.worst.max(d) };
                if self.first.is_none() { self.first = Some(format!("{}: GLSL {x:e}, WGSL {y:e}", what())); }
            }
        }
    }
    fn add(&mut self, other: Tally) {
        self.n += other.n;
        self.bad += other.bad;
        self.worst = self.worst.max(other.worst);
        if self.first.is_none() { self.first = other.first; }
    }
}

/// raw(p), the clip and the shell of one preset, GLSL against WGSL, at points all through the volume.
fn compare_functions(pair: &Pair, points: usize, rng: &mut Rng) -> Tally {
    let p = pair.preset;
    let mut tally = Tally::default();
    for (values, puff) in settings(p, rng) {
        let u = field::uniforms(p, &values, 64, puff);
        let mut g = glsl_machine(&pair.glsl, &u);
        let mut w = wgsl_machine(&pair.wgsl, &u);
        let params = member(w.globals["u"].clone(), "pr");       // what fs does first:  pr = u.pr
        w.set("pr", params);
        let (clip, shell) = (Ex::Id("CLIP".to_string()), Ex::Id("SHELL".to_string()));
        for _ in 0..points {
            let at = [0; 3].map(|_| rng.between(-1.5, 1.5) as f32);
            let label = |what: &str| format!("{} {what} at {at:?} with {values:?}", p.id);
            tally.check(|| label("raw"), &g.call("raw", vec![V(at.to_vec())]), &w.call("raw", vec![V(at.to_vec())]));
            tally.check(|| label("clip"), &g.at_point(&clip, at), &w.at_point(&pair.clip, at));
            tally.check(|| label("shell"), &g.at_point(&shell, at), &w.at_point(&pair.shell, at));
        }
    }
    tally
}

/// What a voxel ends up holding: the GLSL main() against the WGSL fs().
/// The voxels: `voxels` picked at random from a volume 96 across, the 27 at its middle, and the 27 at
/// the middle of a volume 95 across. That last volume has a voxel exactly at the origin, where a
/// symmetric shape has no slope at all and the walk has nothing to go by.
/// Returns the tally, and how many of the voxels hold a distance rather than the -1 or 1 it is clamped to.
fn compare_bake(pair: &Pair, voxels: usize, rng: &mut Rng) -> (Tally, usize) {
    let p = pair.preset;
    let (mut tally, mut near) = (Tally::default(), 0);
    for (values, puff) in settings(p, rng) {
        for (res, random) in [(96u32, voxels), (95, 0)] {
            let u = field::uniforms(p, &values, res, puff);
            let mut g = glsl_machine(&pair.glsl, &u);
            let mut w = wgsl_machine(&pair.wgsl, &u);
            let mut list: Vec<[u32; 3]> = (0..random).map(|_| [0; 3].map(|_| rng.below(res))).collect();
            let c = res / 2;
            for i in c - 1..=c + 1 { for j in c - 1..=c + 1 { for z in c - 1..=c + 1 { list.push([i, j, z]); } } }
            for [i, j, z] in list {
                let centre = V(vec![i as f32 + 0.5, j as f32 + 0.5, 0.5, 1.0]);
                let zc = F(((z as f64 + 0.5) / res as f64) as f32);
                g.set("gl_FragCoord", centre.clone());
                g.set("uZ", zc.clone());
                g.call("main", vec![]);
                let glsl = g.globals["o"].clone();
                let wgsl = w.call("fs", vec![S(BTreeMap::from([("pos".to_string(), centre), ("zc".to_string(), zc)]))]);
                tally.check(|| format!("{} voxel ({i}, {j}, {z}) of {res} with {values:?}, puff {puff}", p.id), &glsl, &wgsl);
                if let V(v) = &glsl { if v[0].abs() < 1.0 { near += 1; } }
            }
        }
    }
    (tally, near)
}

#[test]
fn raw_clip_and_shell_give_the_same_values() {
    let (fams, dump) = (families(), dump());
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut all = Tally::default();
    for pair in pairs(&fams, &dump) {
        let t = compare_functions(&pair, 300, &mut rng);
        println!("{:10} {:12} {:5} values compared, {} differ", pair.family, pair.preset.id, t.n, t.bad);
        all.add(t);
    }
    println!("raw, clip, shell: {} values compared, {} differ, largest difference {:e}", all.n, all.bad, all.worst);
    assert!(all.bad == 0, "{} of {} values differ (largest difference {:e}); the first: {}", all.bad, all.n, all.worst, all.first.unwrap());
    assert_eq!(all.n, 18 * 4 * 300 * 3);
}

#[test]
fn a_whole_voxel_gives_the_same_value() {
    let (fams, dump) = (families(), dump());
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let (mut all, mut near) = (Tally::default(), 0);
    for pair in pairs(&fams, &dump) {
        let (t, n) = compare_bake(&pair, 150, &mut rng);
        println!("{:10} {:12} {:4} voxels compared ({n} of them not clamped to -1 or 1), {} values differ", pair.family, pair.preset.id, t.n / 4, t.bad);
        // the comparison means little if nearly every voxel is clamped to the same -1 or 1
        assert!(n >= 300, "{}: only {n} voxels hold a distance", pair.preset.id);
        near += n;
        all.add(t);
    }
    println!("whole bake: {} voxels compared ({near} not clamped), {} values differ, largest difference {:e}", all.n / 4, all.bad, all.worst);
    assert!(all.bad == 0, "{} of {} values differ (largest difference {:e}); the first: {}", all.bad, all.n, all.worst, all.first.unwrap());
    assert_eq!(all.n, 18 * 4 * (150 + 27 + 27) * 4);
}

#[test]
fn the_slice_triangle_is_the_same() {
    let (fams, dump) = (families(), dump());
    let glsl = parse_glsl(dump["vertex"].as_str().unwrap());
    let recorded = &dump["families"][0]["presets"][0]["bake"];
    let mut n = 0;
    for pair in pairs(&fams, &dump) {
        // the web version's triangle, and where it puts each slice
        let mut g = Machine::new(&glsl);
        let corners: Vec<Val> = (0..3).map(|vi| {
            g.set("gl_VertexID", I(vi));
            g.set("gl_Position", V(vec![0.0; 4]));
            g.call("main", vec![]);
            g.globals["gl_Position"].clone()
        }).collect();
        assert_eq!(corners, [V(vec![-1.0, -1.0, 0.0, 1.0]), V(vec![3.0, -1.0, 0.0, 1.0]), V(vec![-1.0, 3.0, 0.0, 1.0])]);
        for res in [64u32, 96, 192, 256, 320, 384, 512, 768, 896] {
            let u = field::uniforms(pair.preset, &[], res, 0.0);
            let mut w = wgsl_machine(&pair.wgsl, &u);
            for z in 0..res {
                for vi in 0..3 {
                    let out = w.call("vs", vec![I(vi), I(z as i64)]);
                    assert_eq!(member(out.clone(), "pos"), corners[vi as usize], "{}: corner {vi}", pair.preset.id);
                    // the depth the web version works out in 64 bits and hands over as a float
                    assert_eq!(member(out, "zc"), F(((z as f64 + 0.5) / res as f64) as f32), "{}: slice {z} of {res}", pair.preset.id);
                    n += 1;
                }
            }
        }
        // and the depths WebGL was really given for a few slices of the full-size volume
        let res = recorded["res"].as_u64().unwrap() as u32;
        let u = field::uniforms(pair.preset, &[], res, 0.0);
        let mut w = wgsl_machine(&pair.wgsl, &u);
        for zc in recorded["zcBits"].as_array().unwrap() {
            let out = w.call("vs", vec![I(0), I(zc[0].as_i64().unwrap())]);
            assert_eq!(member(out, "zc"), F(f32::from_bits(zc[1].as_u64().unwrap() as u32)), "{}: slice {} of {res}", pair.preset.id, zc[0]);
        }
    }
    println!("slice triangle: {n} corners compared, all the same");
}

// ------------------------------------------------------------------ the interpreter itself
fn glsl_value(src: &str) -> Val {
    let prog = Program::default();
    concrete(Machine::new(&prog).eval(&parse_expr(src, false)))
}
fn wgsl_value(src: &str) -> Val {
    let prog = Program { wgsl: true, ..Default::default() };
    concrete(Machine::new(&prog).eval(&parse_expr(src, true)))
}

#[test]
fn the_interpreter_reads_expressions_properly() {
    for value in [glsl_value, wgsl_value] {
        assert_eq!(value("1.0 - 2.0 * 3.0"), F(-5.0));
        assert_eq!(value("(1.0 - 2.0) * 3.0"), F(-3.0));
        assert_eq!(value("8.0 / 4.0 / 2.0"), F(1.0));
        assert_eq!(value("2.0 - 3.0 - 4.0"), F(-5.0));
        assert_eq!(value("-2.0 * -3.0"), F(6.0));
        assert_eq!(value("2.0 - -3.0"), F(5.0));
        assert_eq!(value("1e-3 * 2e3"), F(1e-3 * 2e3));
        assert_eq!(value("1.0 / 3.0"), F(1.0 / 3.0));
        assert_eq!(value("2.0 > 1.0"), B(true));
        assert_eq!(value("1.5 < 1.0"), B(false));
        assert_eq!(value("(3 << 1) & 2"), I(2));
    }
    // GLSL
    assert_eq!(glsl_value("vec3(1.0, 2.0, 3.0).yzx"), V(vec![2.0, 3.0, 1.0]));
    assert_eq!(glsl_value("vec3(1.0, 2.0, 3.0).xzy"), V(vec![1.0, 3.0, 2.0]));
    assert_eq!(glsl_value("vec2(1.0, -1.0).xyy"), V(vec![1.0, -1.0, -1.0]));
    assert_eq!(glsl_value("vec3(1.0, 2.0, 3.0).z"), F(3.0));
    assert_eq!(glsl_value("vec3(2.0) * vec3(1.0, 2.0, 3.0) + 1.0"), V(vec![3.0, 5.0, 7.0]));
    assert_eq!(glsl_value("2.0 > 1.0 ? 5.0 : 7.0"), F(5.0));
    assert_eq!(glsl_value("2.0 < 1.0 ? 5.0 : 7.0"), F(7.0));
    assert_eq!(glsl_value("2.0 < 1.0 ? 5.0 : 3.0 > 1.0 ? 6.0 : 7.0"), F(6.0));
    assert_eq!(glsl_value("dot(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0))"), F(32.0));
    assert_eq!(glsl_value("length(vec3(3.0, 4.0, 0.0))"), F(5.0));
    assert_eq!(glsl_value("mix(2.0, 4.0, 0.25)"), F(2.5));
    assert_eq!(glsl_value("clamp(5.0, -1.0, 1.0)"), F(1.0));
    assert_eq!(glsl_value("smoothstep(0.0, 2.0, 1.0)"), F(0.5));
    assert_eq!(glsl_value("max(vec3(1.0, -2.0, 3.0), 0.0)"), V(vec![1.0, 0.0, 3.0]));
    assert_eq!(glsl_value("min(1.0, 2.0) - abs(-3.0)"), F(-2.0));
    assert_eq!(glsl_value("pow(2.0, 3.0)"), F(8.0));
    assert_eq!(glsl_value("float(3) * 0.5"), F(1.5));
    // WGSL
    assert_eq!(wgsl_value("vec3f(1.0, 2.0, 3.0).yzx"), V(vec![2.0, 3.0, 1.0]));
    assert_eq!(wgsl_value("select(f32(7), 5.0, 2.0 > 1.0)"), F(5.0));
    assert_eq!(wgsl_value("select(f32(7), 5.0, 2.0 < 1.0)"), F(7.0));
    assert_eq!(wgsl_value("max(vec3f(1.0, -2.0, 3.0), vec3f(0.0))"), V(vec![1.0, 0.0, 3.0]));
    assert_eq!(wgsl_value("mix(f32(2), 4.0, 0.25)"), F(2.5));
    assert_eq!(wgsl_value("f32((3u << 1u) & 2u) + 0.5"), F(2.5));
    // a number written out in WGSL is worked on in 64 bits until it meets a float
    assert_eq!(Machine::new(&Program { wgsl: true, ..Default::default() }).eval(&parse_expr("0.1 + 0.2", true)), A(0.1 + 0.2));
    assert_eq!(wgsl_value("0.1 + 0.2"), F((0.1f64 + 0.2f64) as f32));
    assert_eq!(wgsl_value("f32(1) * 0.1 + 0.2"), F(1.0f32 * 0.1f32 + 0.2f32));
}

/// Two presets worked out by hand, one from each family, so that the interpreter is itself held to
/// something outside it. Their GLSL is run and must give exactly this.
#[test]
#[allow(clippy::approx_constant, clippy::excessive_precision)]      // pi as the shader writes it
fn the_interpreter_gives_the_right_answers() {
    let (fams, dump) = (families(), dump());
    let pairs = pairs(&fams, &dump);
    let mut rng = Rng(0x1234567887654321);

    let gyroid = pairs.iter().find(|p| p.preset.id == "gyroidnet").unwrap();
    let tangle = pairs.iter().find(|p| p.preset.id == "tanglecube").unwrap();
    for _ in 0..200 {
        let at = [0; 3].map(|_| rng.between(-1.5, 1.5) as f32);

        // cells, level, solid, wall, box
        let (cells, level) = (rng.between(0.6, 5.0) as f32, rng.between(-1.2, 1.2) as f32);
        let u = field::uniforms(gyroid.preset, &[cells as f64, level as f64, 1.0, 0.045, 0.0], 64, 0.0);
        let q = at.map(|x| x * 3.14159265f32 * cells);
        let (s, c) = (q.map(f32::sin), q.map(f32::cos));
        let v = (0.0 + s[0] * c[1] + s[1] * c[2] + s[2] * c[0]) - level;
        assert_eq!(glsl_machine(&gyroid.glsl, &u).call("raw", vec![V(at.to_vec())]), F(-v));

        let c0 = rng.between(7.5, 14.5) as f32;
        let u = field::uniforms(tangle.preset, &[c0 as f64], 64, 0.0);
        let a = at.map(|x| (x * 2.55f32) * (x * 2.55f32));
        let v = (0.0 + a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) - 5.0 * (a[0] + a[1] + a[2]) + c0;
        assert_eq!(glsl_machine(&tangle.glsl, &u).call("raw", vec![V(at.to_vec())]), F(v));
    }
}

/// The comparison has teeth: a slip in a translation is noticed, whichever line it is on. Each of these
/// puts one mistake into one preset's WGSL and expects the values to stop agreeing.
#[test]
fn a_wrong_translation_is_caught() {
    let (fams, dump) = (families(), dump());
    let mut pairs = pairs(&fams, &dump);
    let slips: &[(&str, &str, &str)] = &[
        // in a preset's own code
        ("gyroid", "cos(q.yzx)", "cos(q.zxy)"),                                          // a swizzle the wrong way round
        ("gyroid", "select(v, -v, pr.solid > 0.5)", "select(-v, v, pr.solid > 0.5)"),   // select's arguments in GLSL's order
        ("heart", "1.0/3.0", "0.3333333"),                                               // a constant rounded off
        ("barth", "(1.0 + 2.0*ph)", "1.0 + 2.0*ph"),                                     // a lost bracket
        ("kummer", "const R2: f32 = 1.41421356;", "const R2: f32 = 1.414213;"),          // digits dropped
        ("neovius", "pr.box)", "1.0 - pr.box)"),                                         // the clip's switch backwards
        ("schwarzd", "select(pr.wall, 0.0, pr.solid > 0.5)", "select(0.0, pr.wall, pr.solid > 0.5)"),
        // in the walk to the surface, line by line
        ("fischerkoch", "vec3f(xy / u.res", "vec3f(xy.yx / u.res"),
        ("orthocircles", "in.zc", "0.5"),
        ("clebsch", "* u.hb", "* 1.4"),
        ("pretzel", "const e: f32 = 0.0007;", "const e: f32 = 0.0008;"),
        ("chmutov6", "for (var i = 0; i < 8; i++)", "for (var i = 0; i < 7; i++)"),
        ("lidinoid", "raw(x + e*k.yxy)", "raw(x + e*k.xyx)"),                            // the slope sampled at a wrong corner
        ("kummer", "0.25 * (a + b + c + d)", "0.25 * (a + b + c - d)"),
        ("pretzel", "(4.0 * e)", "(2.0 * e)"),
        ("chmutov4", "if (i == 0) { v0 = fv; }", "v0 = fv;"),                            // which side: taken from the last step, not the first
        ("chmutov4", "max(dot(g, g), 1e-12)", "max(dot(g, g), 0.0)"),                    // no guard where there is no slope
        ("iwp", "if (last > 0.12) { st *= 0.12 / last; }", ""),                          // leaping
        ("iwp", "st *= 0.12 / last;", "st *= 0.13 / last;"),
        ("schwarzd", "x -= st;", "x += st;"),
        ("tanglecube", "if (last < 2e-5) { break; }", ""),                               // never stopping early
        ("tanglecube", "mix(1.0, 0.7,", "mix(0.7, 1.0,"),
        ("decocube", "smoothstep(0.06, 0.2, dist)", "smoothstep(0.06, 0.3, dist)"),
        ("barth", "select(dist, -dist, v0 < 0.0)", "select(-dist, dist, v0 < 0.0)"),
        ("schwarzp", "- u.puff", "+ u.puff"),                                            // inflating the wrong way
        ("gyroid", "if (w > 0.0) { sd = abs(sd) - w; }", ""),                            // no wall
        ("decocube", "smax(sd,", "smin(sd,"),
        ("clebsch", "), 0.03), -1.0, 1.0)", "), 0.04), -1.0, 1.0)"),
        ("orthocircles", "-1.0, 1.0), 0.0, 0.0, 1.0)", "-1.0, 0.9), 0.0, 0.0, 1.0)"),
    ];
    for (id, right, wrong) in slips {
        let pair = pairs.iter_mut().find(|p| p.preset.id == *id).unwrap();
        let good = field::bake_shader(pair.preset);
        assert_eq!(good.matches(right).count(), 1, "{id}: expected {right:?} in the shader just once");
        pair.wgsl = parse_wgsl(&good.replace(right, wrong));
        if let Shape::Field { clip, shell, .. } = &pair.preset.shape {
            pair.clip = parse_expr(&clip.replace(right, wrong), true);
            pair.shell = parse_expr(&shell.replace(right, wrong), true);
        }
        let mut rng = Rng(0x0123456789abcdef);
        let functions = compare_functions(pair, 100, &mut rng);
        let (bake, _) = compare_bake(pair, 60, &mut rng);
        println!("{id:12} {:4} of {} function values, {:3} of {} voxel values differ with {wrong:?} for {right:?}", functions.bad, functions.n, bake.bad, bake.n);
        assert!(functions.bad + bake.bad > 0, "{id}: {wrong:?} in place of {right:?} went unnoticed");

        // and with the slip taken out again, they agree
        pair.wgsl = parse_wgsl(&good);
        if let Shape::Field { clip, shell, .. } = &pair.preset.shape {
            pair.clip = parse_expr(clip, true);
            pair.shell = parse_expr(shell, true);
        }
        let (bake, _) = compare_bake(pair, 20, &mut rng);
        assert_eq!(bake.bad, 0, "{id}");
    }
    println!("{} slips, each one noticed", slips.len());
}
