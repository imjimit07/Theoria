//! Recursive-descent parser: tokens to CST.
//!
//! Fail-fast: the first error stops parsing. Each `parse_*` method
//! captures its start span before descending and merges it with the last
//! consumed token's span, so every CST node covers exactly the source it
//! was built from.

use theoria_syntax::{
    BinOp, ErrorKind, Expr, FieldDef, FunctionDef, Hypothesis, IfExpr, ImportDecl, ImportName,
    LamParam, MatchArm, MatchExpr, Module, ModuleDecl, Param, Path, PathSegment, Pattern,
    PatternArg, ProofBlock, ProofStep, ProofStepKind, Span, Stmt, StructureDef, SyntaxError,
    TheoremDef, Token, TokenKind, UnOp,
};

/// Parse a complete `.theoria` source file into a [`Module`].
///
/// On success, the returned `Module`'s `span` covers the whole source.
///
/// On failure, returns at least one [`SyntaxError`]. This delivery
/// returns exactly one error (fail-fast); error recovery is a later
/// addition. The `Vec` return type commits the interface to a
/// multi-error future without changing the call site.
pub fn parse_module(source: &str) -> Result<Module, Vec<SyntaxError>> {
    let tokens = crate::lexer::lex(source).map_err(|e| vec![e])?;
    let mut parser = Parser::new(&tokens, source);
    parser.parse_module_inner().map_err(|e| vec![e])
}

/// The parser: a cursor over the token stream plus the source text.
pub struct Parser<'a> {
    tokens: &'a [Token],
    source: &'a str,
    pos: usize,
    prev_span: Span,
}

impl<'a> Parser<'a> {
    /// Construct a parser over an already-lexed token stream.
    #[must_use]
    pub fn new(tokens: &'a [Token], source: &'a str) -> Self {
        Parser {
            tokens,
            source,
            pos: 0,
            prev_span: Span::empty(0),
        }
    }

    // -----------------------------------------------------------------
    // Cursor primitives
    // -----------------------------------------------------------------

    fn peek_token(&self) -> Token {
        self.tokens.get(self.pos).copied().unwrap_or(Token::new(
            TokenKind::Eof,
            Span::empty(self.source.len() as u32),
        ))
    }

    fn peek(&self) -> TokenKind {
        self.peek_token().kind
    }

    #[allow(dead_code)]
    fn peek_at(&self, offset: usize) -> TokenKind {
        self.tokens
            .get(self.pos + offset)
            .map_or(TokenKind::Eof, |t| t.kind)
    }

    fn peek_span(&self) -> Span {
        self.peek_token().span
    }

    fn advance(&mut self) -> Token {
        let t = self.peek_token();
        self.prev_span = t.span;
        if t.kind != TokenKind::Eof {
            self.pos += 1;
        }
        t
    }

    fn err_here(&self, kind: ErrorKind, message: String) -> SyntaxError {
        SyntaxError::new(self.peek_span(), kind, message)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, SyntaxError> {
        let t = self.peek_token();
        if t.kind == kind {
            Ok(self.advance())
        } else if t.kind == TokenKind::Eof {
            Err(SyntaxError::new(
                t.span,
                ErrorKind::UnexpectedEof,
                format!("unexpected end of input; expected {kind:?}"),
            ))
        } else {
            Err(SyntaxError::new(
                t.span,
                ErrorKind::UnexpectedToken,
                format!("expected {kind:?}, found {actual:?}", actual = t.kind),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), SyntaxError> {
        let t = self.peek_token();
        if t.kind == TokenKind::Ident {
            self.advance();
            Ok((self.slice(t.span).to_string(), t.span))
        } else if t.kind == TokenKind::Eof {
            Err(SyntaxError::new(
                t.span,
                ErrorKind::UnexpectedEof,
                "unexpected end of input; expected an identifier".to_string(),
            ))
        } else {
            Err(SyntaxError::new(
                t.span,
                ErrorKind::UnexpectedToken,
                format!("expected an identifier, found {actual:?}", actual = t.kind),
            ))
        }
    }

    /// `true` iff the next token is the identifier `text`.
    fn at_keyword(&self, text: &str) -> bool {
        let t = self.peek_token();
        t.kind == TokenKind::Ident && self.slice(t.span) == text
    }

    fn expect_keyword(&mut self, text: &str) -> Result<Token, SyntaxError> {
        let t = self.peek_token();
        if t.kind == TokenKind::Ident && self.slice(t.span) == text {
            Ok(self.advance())
        } else if t.kind == TokenKind::Eof {
            Err(SyntaxError::new(
                t.span,
                ErrorKind::UnexpectedEof,
                format!("unexpected end of input; expected keyword `{text}`"),
            ))
        } else {
            Err(SyntaxError::new(
                t.span,
                ErrorKind::UnexpectedToken,
                format!("expected keyword `{text}`"),
            ))
        }
    }

    fn expect_binding_ident(&mut self, what: &str) -> Result<(String, Span), SyntaxError> {
        let (text, span) = self.expect_ident()?;
        Ok((Self::check_binding_name(text, span, what)?, span))
    }

    fn slice(&self, span: Span) -> &str {
        &self.source[span.start as usize..span.end as usize]
    }

    // -----------------------------------------------------------------
    // Module-level productions
    // -----------------------------------------------------------------

    fn parse_module_inner(&mut self) -> Result<Module, SyntaxError> {
        let mut decl = None;
        let mut imports = Vec::new();
        let mut structures = Vec::new();
        let mut functions = Vec::new();
        let mut theorems = Vec::new();
        loop {
            match self.peek() {
                TokenKind::Eof => break,
                TokenKind::Newline => {
                    self.advance();
                }
                TokenKind::Ident if self.at_keyword("Module") => {
                    if decl.is_some() {
                        return Err(self.err_here(
                            ErrorKind::UnexpectedToken,
                            "duplicate Module declaration".to_string(),
                        ));
                    }
                    decl = Some(self.parse_module_decl()?);
                }
                TokenKind::Ident if self.at_keyword("Import") => {
                    imports.push(self.parse_import_decl()?);
                }
                TokenKind::Ident if self.at_keyword("Structure") => {
                    structures.push(self.parse_structure_def()?);
                }
                TokenKind::Ident if self.at_keyword("Function") => {
                    functions.push(self.parse_function_def()?);
                }
                TokenKind::Ident if self.at_keyword("Theorem") => {
                    theorems.push(self.parse_theorem_def()?);
                }
                TokenKind::Ident => {
                    let (text, span) = self.expect_ident()?;
                    return Err(SyntaxError::new(
                        span,
                        ErrorKind::NotImplemented,
                        format!(
                            "`{text}` is not yet implemented \
                             (only Module, Import, Structure, Function, and Theorem are supported in this delivery)"
                        ),
                    ));
                }
                _ => {
                    return Err(self.err_here(
                        ErrorKind::UnexpectedToken,
                        "expected a top-level Module, Import, Structure, Function, or Theorem declaration"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(Module {
            decl,
            imports,
            structures,
            functions,
            theorems,
            span: Span::new(0, self.source.len() as u32),
        })
    }

    fn parse_module_decl(&mut self) -> Result<ModuleDecl, SyntaxError> {
        let kw = self.expect_keyword("Module")?;
        let path = self.parse_path()?;
        self.expect(TokenKind::Newline)?;
        let span = Span::covering(kw.span, self.prev_span);
        Ok(ModuleDecl { path, span })
    }

    fn parse_import_decl(&mut self) -> Result<ImportDecl, SyntaxError> {
        let kw = self.expect_keyword("Import")?;
        let path = self.parse_path()?;
        let mut names = None;
        if self.peek() == TokenKind::LParen {
            self.advance();
            let mut list = Vec::new();
            if self.peek() != TokenKind::RParen {
                loop {
                    let (text, span) = self.expect_ident()?;
                    list.push(ImportName { text, span });
                    if self.peek() == TokenKind::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;
            names = Some(list);
        }
        self.expect(TokenKind::Newline)?;
        let span = Span::covering(kw.span, self.prev_span);
        Ok(ImportDecl { path, names, span })
    }

    fn parse_function_def(&mut self) -> Result<FunctionDef, SyntaxError> {
        let kw = self.expect_keyword("Function")?;
        let (name, name_span) = self.expect_binding_ident("a function name")?;
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if self.peek() != TokenKind::RParen {
            loop {
                params.push(self.parse_param()?);
                if self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen)?;
        let mut return_ty = None;
        if self.peek() == TokenKind::Arrow {
            self.advance();
            return_ty = Some(self.parse_expr()?);
        }
        self.expect(TokenKind::Colon)?;
        self.expect(TokenKind::Newline)?;
        if self.peek() != TokenKind::Indent {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected an indented block after `:`".to_string(),
            ));
        }
        self.advance();
        if self.peek() == TokenKind::Dedent {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "empty function body".to_string(),
            ));
        }
        let mut body = Vec::new();
        // Zero or more `let` statements...
        loop {
            match self.peek() {
                TokenKind::Dedent | TokenKind::Eof => break,
                // Blank lines inside a body are skipped, as at top level.
                TokenKind::Newline => {
                    self.advance();
                }
                _ if self.at_keyword("let") => body.push(self.parse_let_stmt()?),
                _ => break,
            }
        }
        // ...followed by exactly one `return`. Anything else here —
        // including a bare `Dedent` (a body of only `let`s) — is an
        // error.
        match self.peek() {
            TokenKind::Dedent | TokenKind::Eof => {
                return Err(
                    self.err_here(ErrorKind::UnexpectedToken, "expected `return`".to_string())
                );
            }
            _ => body.push(self.parse_stmt()?),
        }
        // Blank lines between the `return` and the dedent are skipped.
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        self.expect(TokenKind::Dedent)?;
        let span = Span::covering(kw.span, self.prev_span);
        Ok(FunctionDef {
            name,
            name_span,
            params,
            return_ty,
            body,
            span,
        })
    }

    fn parse_param(&mut self) -> Result<Param, SyntaxError> {
        let (name, name_span) = self.expect_binding_ident("a parameter name")?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_expr()?;
        let span = Span::covering(name_span, self.prev_span);
        Ok(Param {
            name,
            name_span,
            ty,
            span,
        })
    }

    /// `Structure Name[(P₁ : T₁, ...)]:` followed by an indented block of
    /// `name : ty` field declarations.
    ///
    /// The parameter list is optional; a parameterless structure omits the
    /// parentheses entirely. At least one field is required.
    fn parse_structure_def(&mut self) -> Result<StructureDef, SyntaxError> {
        let kw = self.expect_keyword("Structure")?;
        let (name, name_span) = self.expect_binding_ident("a structure name")?;
        let mut params = Vec::new();
        if self.peek() == TokenKind::LParen {
            self.advance();
            if self.peek() != TokenKind::RParen {
                loop {
                    params.push(self.parse_param()?);
                    if self.peek() == TokenKind::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;
        }
        self.expect(TokenKind::Colon)?;
        self.expect(TokenKind::Newline)?;
        if self.peek() != TokenKind::Indent {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected an indented block of field declarations after `Structure ...:`"
                    .to_string(),
            ));
        }
        self.advance();
        let mut fields = Vec::new();
        loop {
            // Blank lines between fields are skipped, as at top level.
            while self.peek() == TokenKind::Newline {
                self.advance();
            }
            match self.peek() {
                TokenKind::Dedent | TokenKind::Eof => break,
                TokenKind::Ident => fields.push(self.parse_field_def()?),
                _ => {
                    return Err(self.err_here(
                        ErrorKind::UnexpectedToken,
                        "expected a field declaration or the end of the block".to_string(),
                    ));
                }
            }
        }
        if fields.is_empty() {
            return Err(SyntaxError::new(
                name_span,
                ErrorKind::UnexpectedToken,
                format!("structure `{name}` must declare at least one field"),
            ));
        }
        // Blank lines between the last field and the dedent are skipped.
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        self.expect(TokenKind::Dedent)?;
        let span = Span::covering(kw.span, self.prev_span);
        Ok(StructureDef {
            name,
            name_span,
            params,
            fields,
            span,
        })
    }

    /// `Theorem name:` followed by optional `Given:` and `Assume:`
    /// sections, a `Show:` line, a `Proof:` line, and `QED`.
    ///
    /// Section shape:
    ///
    /// * `Given:` is a block of `name : type` bindings (zero or more).
    /// * `Assume:` is a block of `name : proposition` bindings (zero or
    ///   more).
    /// * `Show:` is a single-line goal expression.
    /// * `Proof:` is a single-line proof expression.
    /// * `QED` terminates the theorem.
    ///
    /// Multi-line `Show:` and `Proof:` bodies, and the plan's numbered
    /// proof steps, are deferred.
    fn parse_theorem_def(&mut self) -> Result<TheoremDef, SyntaxError> {
        let kw = self.expect_keyword("Theorem")?;
        let (name, name_span) = self.expect_binding_ident("a theorem name")?;
        self.expect(TokenKind::Colon)?;
        self.expect(TokenKind::Newline)?;
        self.expect(TokenKind::Indent)?;
        while self.peek() == TokenKind::Newline {
            self.advance();
        }

        // Given: optional block of params.
        let mut given = Vec::new();
        if self.at_keyword("Given") {
            self.advance();
            self.expect(TokenKind::Colon)?;
            self.expect(TokenKind::Newline)?;
            self.expect(TokenKind::Indent)?;
            loop {
                while self.peek() == TokenKind::Newline {
                    self.advance();
                }
                match self.peek() {
                    TokenKind::Dedent | TokenKind::Eof => break,
                    TokenKind::Ident => given.push(self.parse_param()?),
                    _ => {
                        return Err(self.err_here(
                            ErrorKind::UnexpectedToken,
                            "expected a `name : type` binding in the Given block".to_string(),
                        ));
                    }
                }
            }
            self.expect(TokenKind::Dedent)?;
            while self.peek() == TokenKind::Newline {
                self.advance();
            }
        }

        // Assume: optional block of hypotheses.
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        let mut assume = Vec::new();
        if self.at_keyword("Assume") {
            self.advance();
            self.expect(TokenKind::Colon)?;
            self.expect(TokenKind::Newline)?;
            self.expect(TokenKind::Indent)?;
            loop {
                while self.peek() == TokenKind::Newline {
                    self.advance();
                }
                match self.peek() {
                    TokenKind::Dedent | TokenKind::Eof => break,
                    TokenKind::Ident => assume.push(self.parse_hypothesis()?),
                    _ => {
                        return Err(self.err_here(
                            ErrorKind::UnexpectedToken,
                            "expected a `name : proposition` binding in the Assume block"
                                .to_string(),
                        ));
                    }
                }
            }
            self.expect(TokenKind::Dedent)?;
            while self.peek() == TokenKind::Newline {
                self.advance();
            }
        }

        while self.peek() == TokenKind::Newline {
            self.advance();
        }

        // Show: required, single-line.
        self.expect_keyword("Show")?;
        self.expect(TokenKind::Colon)?;
        let show = self.parse_expr()?;
        self.expect(TokenKind::Newline)?;

        // Proof: required. Either a single-line term or an indented
        // block of numbered steps.
        self.expect_keyword("Proof")?;
        self.expect(TokenKind::Colon)?;
        let proof = if self.peek() == TokenKind::Newline {
            self.advance();
            self.parse_proof_steps()?
        } else {
            let e = self.parse_expr()?;
            self.expect(TokenKind::Newline)?;
            ProofBlock::Term(e)
        };

        // QED: required.
        self.expect_keyword("QED")?;
        let end = self.expect(TokenKind::Newline)?.span.end;
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        self.expect(TokenKind::Dedent)?;
        let span = Span::covering(kw.span, Span::new(end, end));

        Ok(TheoremDef {
            name,
            name_span,
            given,
            assume,
            show,
            proof,
            span,
        })
    }

    fn parse_hypothesis(&mut self) -> Result<Hypothesis, SyntaxError> {
        let (name, name_span) = self.expect_binding_ident("a hypothesis name")?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_expr()?;
        let end = self.expect(TokenKind::Newline)?.span.end;
        Ok(Hypothesis {
            name,
            name_span,
            ty,
            span: Span::new(name_span.start, end),
        })
    }

    /// Parse the indented block of numbered steps following `Proof:`.
    ///
    /// Enforces at least one step and strictly sequential numbering
    /// starting at `1`.
    fn parse_proof_steps(&mut self) -> Result<ProofBlock, SyntaxError> {
        if self.peek() != TokenKind::Indent {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected an indented block of numbered proof steps after `Proof:`".to_string(),
            ));
        }
        self.advance();
        let mut steps: Vec<ProofStep> = Vec::new();
        let mut expected: u32 = 1;
        loop {
            // Blank lines between steps are allowed.
            while self.peek() == TokenKind::Newline {
                self.advance();
            }
            match self.peek() {
                TokenKind::Dedent | TokenKind::Eof => break,
                TokenKind::Nat => {
                    let step = self.parse_proof_step()?;
                    if step.number != expected {
                        return Err(SyntaxError::new(
                            step.number_span,
                            ErrorKind::UnexpectedToken,
                            format!("expected step number {expected}, found {}", step.number),
                        ));
                    }
                    expected += 1;
                    steps.push(step);
                }
                _ => {
                    return Err(self.err_here(
                        ErrorKind::UnexpectedToken,
                        "expected a numbered proof step (`N. Let …` / `N. Have …` / `N. Exact …`)"
                            .to_string(),
                    ));
                }
            }
        }
        if steps.is_empty() {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "a `Proof:` block must contain at least one step".to_string(),
            ));
        }
        // Blank lines before the closing dedent.
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        self.expect(TokenKind::Dedent)?;
        Ok(ProofBlock::Steps(steps))
    }

    /// Parse a single numbered step.
    ///
    /// Grammar:
    ///
    /// ```text
    /// Step     ::= Nat "." StepKind [ "[" justification "]" ] Newline
    /// StepKind ::= "Let"   Ident [ ":" Expr ] "=" Expr
    ///            | "Have"  Ident ":" Expr   "=" Expr
    ///            | "Exact" Expr
    /// ```
    fn parse_proof_step(&mut self) -> Result<ProofStep, SyntaxError> {
        let num_tok = self.advance(); // TokenKind::Nat
        let text = self.slice(num_tok.span);
        let number: u32 = text.parse().map_err(|_| {
            SyntaxError::new(
                num_tok.span,
                ErrorKind::InvalidLiteral,
                "step number out of range".to_string(),
            )
        })?;
        self.expect(TokenKind::Dot)?;
        let kind = self.parse_proof_step_kind()?;
        let justification = if self.peek() == TokenKind::LBracket {
            Some(self.parse_justification()?)
        } else {
            None
        };
        let end = self.expect(TokenKind::Newline)?.span.end;
        Ok(ProofStep {
            number,
            number_span: num_tok.span,
            kind,
            justification,
            span: Span::new(num_tok.span.start, end),
        })
    }

    /// Parse the keyword and payload of a proof step.
    fn parse_proof_step_kind(&mut self) -> Result<ProofStepKind, SyntaxError> {
        if self.at_keyword("Let") {
            self.advance();
            let (name, name_span) = self.expect_binding_ident("a step binding name")?;
            // Optional `: ty` annotation.
            let ty = if self.peek() == TokenKind::Colon {
                self.advance();
                Some(self.parse_expr()?)
            } else {
                None
            };
            self.expect(TokenKind::Eq)?;
            let value = self.parse_expr()?;
            Ok(ProofStepKind::Let {
                name,
                name_span,
                ty,
                value,
            })
        } else if self.at_keyword("Have") {
            self.advance();
            let (name, name_span) = self.expect_binding_ident("a step binding name")?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_expr()?;
            self.expect(TokenKind::Eq)?;
            let proof = self.parse_expr()?;
            Ok(ProofStepKind::Have {
                name,
                name_span,
                ty,
                proof,
            })
        } else if self.at_keyword("Exact") {
            self.advance();
            let term = self.parse_expr()?;
            Ok(ProofStepKind::Exact { term })
        } else if self.at_keyword("From") {
            self.advance();
            let (name, name_span) = self.expect_binding_ident("a hypothesis name")?;
            Ok(ProofStepKind::From { name, name_span })
        } else {
            Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected `Let`, `Have`, `Exact`, or `From` at the start of a proof step"
                    .to_string(),
            ))
        }
    }

    /// Parse `[ arbitrary text until the closing bracket ]`.
    ///
    /// The text is retained verbatim; it has no elaboration effect. The
    /// bracket depth is tracked so nested `[` characters (which should
    /// not appear in practice) do not terminate the justification
    /// early.
    #[allow(unused_assignments)]
    fn parse_justification(&mut self) -> Result<String, SyntaxError> {
        let open = self.expect(TokenKind::LBracket)?;
        let start = open.span.end as usize;
        let mut depth: usize = 1;
        let mut text_end = 0usize;
        let mut found = false;
        loop {
            match self.peek() {
                TokenKind::LBracket => {
                    depth += 1;
                    self.advance();
                }
                TokenKind::RBracket => {
                    depth -= 1;
                    let close = self.advance();
                    if depth == 0 {
                        text_end = close.span.start as usize;
                        found = true;
                        break;
                    }
                }
                TokenKind::Eof => {
                    return Err(SyntaxError::new(
                        open.span,
                        ErrorKind::UnexpectedEof,
                        "unterminated `[justification]`".to_string(),
                    ));
                }
                _ => {
                    self.advance();
                }
            }
        }
        debug_assert!(found, "loop breaks with text_end set");
        Ok(self.source[start..text_end].trim().to_string())
    }

    fn parse_field_def(&mut self) -> Result<FieldDef, SyntaxError> {
        let (name, name_span) = self.expect_binding_ident("a field name")?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_expr()?;
        let end = self.expect(TokenKind::Newline)?.span.end;
        Ok(FieldDef {
            name,
            name_span,
            ty,
            span: Span::new(name_span.start, end),
        })
    }

    fn parse_path(&mut self) -> Result<Path, SyntaxError> {
        let (text, span) = self.expect_ident()?;
        let mut segments = vec![PathSegment { text, span }];
        let mut full = span;
        while self.peek() == TokenKind::Dot {
            self.advance();
            let (text, span) = self.expect_ident()?;
            full = Span::covering(full, span);
            segments.push(PathSegment { text, span });
        }
        Ok(Path {
            segments,
            span: full,
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        let kw = self.expect_keyword("return")?;
        let value = self.parse_expr()?;
        // A single-line body ends in a Newline. A block-valued body
        // (currently `match`) ends at a Dedent: its last arm consumed
        // the line ending, so the Dedent is left for the caller.
        loop {
            match self.peek() {
                TokenKind::Newline => {
                    self.advance();
                }
                TokenKind::Dedent => break,
                _ => {
                    return Err(
                        self.err_here(ErrorKind::UnexpectedToken, "expected Newline".to_string())
                    );
                }
            }
        }
        let span = Span::covering(kw.span, self.prev_span);
        Ok(Stmt::Return { value, span })
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        let kw = self.expect_keyword("let")?;
        let (name, name_span) = self.expect_binding_ident("a let binding name")?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_expr()?;
        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        // As in `parse_stmt`: a block value ends wherever the block
        // ends, so a trailing Newline is consumed only if present. The
        // caller validates whatever follows.
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        let span = Span::covering(kw.span, self.prev_span);
        Ok(Stmt::Let {
            name,
            name_span,
            ty,
            value,
            span,
        })
    }

    // -----------------------------------------------------------------
    // Expressions (precedence climbing)
    // -----------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, SyntaxError> {
        if self.at_keyword("if") {
            self.parse_if()
        } else if self.at_keyword("match") {
            self.parse_match()
        } else {
            self.parse_arrow()
        }
    }

    /// `match scrutinee: case P => e ...` with arms indented one level
    /// deeper than `match`.
    ///
    /// The scrutinee is a full expression terminated by `:`, so compound
    /// scrutinees (`match n + 1: ...`, `match if c then a else b: ...`)
    /// need no parens. Each arm body is a full expression, so nested
    /// `if`/`match` in bodies works naturally.
    fn parse_match(&mut self) -> Result<Expr, SyntaxError> {
        let kw = self.expect_keyword("match")?;
        let scrutinee = self.parse_expr()?;
        self.expect(TokenKind::Colon)?;
        self.expect(TokenKind::Newline)?;
        if self.peek() != TokenKind::Indent {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected an indented block of `case` arms after `match ...:`".to_string(),
            ));
        }
        self.advance();
        let mut arms = Vec::new();
        loop {
            match self.peek() {
                TokenKind::Dedent | TokenKind::Eof => break,
                // Blank lines between arms are skipped.
                TokenKind::Newline => {
                    self.advance();
                }
                _ if self.at_keyword("case") => arms.push(self.parse_match_arm()?),
                _ => {
                    return Err(
                        self.err_here(ErrorKind::UnexpectedToken, "expected `case`".to_string())
                    );
                }
            }
        }
        if arms.is_empty() {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected at least one `case` arm".to_string(),
            ));
        }
        // Blank lines between the last arm and the dedent are skipped.
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
        self.expect(TokenKind::Dedent)?;
        let span = Span::covering(kw.span, self.prev_span);
        Ok(Expr::Match(MatchExpr {
            scrutinee: Box::new(scrutinee),
            arms,
            span,
        }))
    }

    fn parse_match_arm(&mut self) -> Result<MatchArm, SyntaxError> {
        let kw = self.expect_keyword("case")?;
        let pattern = self.parse_pattern()?;
        // `=>` lexes as two tokens (`Eq` then `Gt`).
        self.expect(TokenKind::Eq)?;
        self.expect(TokenKind::Gt)?;
        let body = self.parse_expr()?;
        // A single-line body ends in a Newline. A block-valued body
        // (`match`, or an `if` with a `match` branch) ends at a Dedent:
        // the inner block consumed the line ending, so the Dedent is
        // left for the caller.
        if self.peek() == TokenKind::Newline {
            self.advance();
        } else if !matches!(body, Expr::Match(_) | Expr::If(_)) {
            return Err(self.err_here(ErrorKind::UnexpectedToken, "expected Newline".to_string()));
        }
        let span = Span::covering(kw.span, self.prev_span);
        Ok(MatchArm {
            pattern,
            body,
            span,
        })
    }

    /// Parse one flat pattern.
    ///
    /// A bare identifier is a variable binding (constructor names are
    /// always dotted in the current fragment, so an unqualified name
    /// cannot be a constructor). A dotted name must be a constructor;
    /// anything dotted inside an argument list is a nested pattern and
    /// is rejected with `NotImplemented`.
    ///
    /// Constructor arguments come in two shapes: parenthesised
    /// (`Nat.succ(k)`, `Pair(fst, snd)`) or space-separated
    /// (`Nat.succ k`). Both end where the `=>` begins.
    fn parse_pattern(&mut self) -> Result<Pattern, SyntaxError> {
        let (text, span) = self.expect_ident()?;
        if text == "_" {
            return Ok(Pattern::Wildcard(span));
        }
        // A bare identifier is a variable binding — unless it is
        // immediately followed by `(`, in which case it is a
        // constructor with a parenthesised argument list. (Constructor
        // names in the prelude are always dotted, but the parser is
        // name-agnostic: `Pair(fst, snd)` parses as a constructor even
        // though `Pair` does not exist yet.)
        if self.peek() != TokenKind::Dot && self.peek() != TokenKind::LParen {
            return Ok(Pattern::Var {
                name: Self::check_binding_name(text, span, "a pattern variable")?,
                name_span: span,
            });
        }
        // Constructor: dotted name, then optional argument list.
        let mut full = text;
        let mut end = span;
        while self.peek() == TokenKind::Dot {
            self.advance();
            let (seg, seg_span) = self.expect_ident()?;
            full.push('.');
            full.push_str(&seg);
            end = seg_span;
        }
        let mut args = Vec::new();
        if self.peek() == TokenKind::LParen {
            self.advance();
            if self.peek() != TokenKind::RParen {
                loop {
                    args.push(self.parse_pattern_arg()?);
                    if self.peek() == TokenKind::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;
            end = self.prev_span;
        } else {
            // Space-separated field patterns: `Nat.succ k`,
            // `Pair fst snd`. Each must be a bare identifier; the first
            // non-identifier (normally the `=>`) ends the list.
            while self.peek() == TokenKind::Ident {
                args.push(self.parse_pattern_arg()?);
            }
        }
        let span = Span::new(span.start, end.end);
        Ok(Pattern::Constructor {
            name: full,
            name_span: span,
            args,
            span,
        })
    }

    fn parse_pattern_arg(&mut self) -> Result<PatternArg, SyntaxError> {
        let (text, span) = self.expect_ident()?;
        if text == "_" {
            return Ok(PatternArg::Wildcard(span));
        }
        if self.peek() == TokenKind::Dot {
            return Err(SyntaxError::new(
                span,
                ErrorKind::NotImplemented,
                "nested patterns are not supported in this delivery".to_string(),
            ));
        }
        let name = Self::check_binding_name(text, span, "a pattern variable")?;
        Ok(PatternArg::Var {
            name,
            name_span: span,
        })
    }

    /// Reject reserved words in binding positions (parameters, `let` names,
    /// pattern variables). Contextual keywords stay usable as ordinary
    /// expressions; they just cannot bind.
    fn check_binding_name(text: String, span: Span, what: &str) -> Result<String, SyntaxError> {
        const RESERVED: &[&str] = &[
            "Module",
            "Import",
            "Structure",
            "Function",
            "Theorem",
            "if",
            "then",
            "else",
            "let",
            "return",
            "match",
            "case",
            "Given",
            "Assume",
            "Show",
            "Proof",
            "QED",
        ];
        if RESERVED.contains(&text.as_str()) {
            return Err(SyntaxError::new(
                span,
                ErrorKind::UnexpectedToken,
                format!("`{text}` is reserved and cannot be used as {what}"),
            ));
        }
        Ok(text)
    }
    /// `if cond then a else b`.
    ///
    /// The branches extend as far right as possible; `then` and `else`
    /// delimit the condition and the then-branch, so no precedence table
    /// change is needed. Nested `if`s work because the branches re-enter
    /// [`parse_expr`](Parser::parse_expr).
    fn parse_if(&mut self) -> Result<Expr, SyntaxError> {
        let kw = self.expect_keyword("if")?;
        let cond = self.parse_expr()?;
        self.expect_keyword("then")?;
        let then_branch = self.parse_expr()?;
        self.expect_keyword("else")?;
        let else_branch = self.parse_expr()?;
        let span = Span::covering(kw.span, self.prev_span);
        Ok(Expr::If(IfExpr {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
            span,
        }))
    }

    /// Level 1: `->`, right-associative.
    fn parse_arrow(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span();
        let lhs = self.parse_cmp()?;
        if self.peek() == TokenKind::Arrow {
            self.advance();
            let rhs = self.parse_arrow()?;
            let span = Span::covering(start, self.prev_span);
            Ok(Expr::BinOp {
                lhs: Box::new(lhs),
                op: BinOp::Arrow,
                rhs: Box::new(rhs),
                span,
            })
        } else {
            Ok(lhs)
        }
    }

    /// Level 2: comparisons, non-associative.
    fn parse_cmp(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span();
        let lhs = self.parse_add()?;
        let op = match self.peek() {
            TokenKind::EqEq => BinOp::Eq,
            TokenKind::BangEq => BinOp::Ne,
            TokenKind::Lt => BinOp::Lt,
            TokenKind::LtEq => BinOp::Le,
            TokenKind::Gt => BinOp::Gt,
            TokenKind::GtEq => BinOp::Ge,
            _ => return Ok(lhs),
        };
        self.advance();
        let rhs = self.parse_add()?;
        if matches!(
            self.peek(),
            TokenKind::EqEq
                | TokenKind::BangEq
                | TokenKind::Lt
                | TokenKind::LtEq
                | TokenKind::Gt
                | TokenKind::GtEq
        ) {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "comparison operators cannot be chained; write the conjunction explicitly"
                    .to_string(),
            ));
        }
        let span = Span::covering(start, self.prev_span);
        Ok(Expr::BinOp {
            lhs: Box::new(lhs),
            op,
            rhs: Box::new(rhs),
            span,
        })
    }

    /// Level 3: `+` `-`, left-associative.
    fn parse_add(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span();
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_mul()?;
            let span = Span::covering(start, self.prev_span);
            lhs = Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 4: `*` `/`, left-associative.
    fn parse_mul(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span();
        let mut lhs = self.parse_pow()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_pow()?;
            let span = Span::covering(start, self.prev_span);
            lhs = Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    /// Level 5: `^`, right-associative.
    fn parse_pow(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span();
        let base = self.parse_postfix()?;
        if self.peek() == TokenKind::Caret {
            self.advance();
            let exp = self.parse_pow()?;
            let span = Span::covering(start, self.prev_span);
            Ok(Expr::BinOp {
                lhs: Box::new(base),
                op: BinOp::Pow,
                rhs: Box::new(exp),
                span,
            })
        } else {
            Ok(base)
        }
    }

    /// Level 6: postfix `.` `[]` `()`, left-associative.
    fn parse_postfix(&mut self) -> Result<Expr, SyntaxError> {
        let mut expr = self.parse_atom()?;
        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.advance();
                    let (field, field_span) = self.expect_ident()?;
                    let span = Span::covering(expr_span(&expr), field_span);
                    expr = Expr::Field {
                        obj: Box::new(expr),
                        field,
                        field_span,
                        span,
                    };
                }
                TokenKind::LBracket => {
                    // Try to parse `obj[index]`; if the bracketed content
                    // is not a single valid `Expr` that ends at `]`
                    // (e.g. `[by hypothesis]` which is justification text,
                    // not an index), backtrack and leave `[` for the
                    // caller (proof-step justification).
                    let saved_pos = self.pos;
                    let saved_prev = self.prev_span;
                    let open = self.advance();
                    let index_res = (|| -> Result<Expr, SyntaxError> {
                        let idx = self.parse_expr()?;
                        self.expect(TokenKind::RBracket)?;
                        Ok(idx)
                    })();
                    match index_res {
                        Ok(index) => {
                            let span = Span::covering(open.span, self.prev_span);
                            expr = Expr::Index {
                                obj: Box::new(expr),
                                index: Box::new(index),
                                span,
                            };
                        }
                        Err(_) => {
                            // Not a valid Index — restore and stop postfix.
                            self.pos = saved_pos;
                            self.prev_span = saved_prev;
                            break;
                        }
                    }
                }
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if self.peek() != TokenKind::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek() == TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    let span = Span::covering(expr_span(&expr), self.prev_span);
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Atoms, plus unary minus.
    ///
    /// The operand of `-` parses at [`Pow`](Parser::parse_pow) level (not
    /// atom level), so `-x^2` parses as `-(x^2)`: the exponent binds
    /// tighter than the sign.
    fn parse_atom(&mut self) -> Result<Expr, SyntaxError> {
        match self.peek() {
            TokenKind::Ident => {
                let (text, span) = self.expect_ident()?;
                Ok(Expr::Ident { text, span })
            }
            TokenKind::Nat => {
                let t = self.advance();
                let text = self.slice(t.span);
                let value = text.parse::<u64>().map_err(|_| {
                    SyntaxError::new(
                        t.span,
                        ErrorKind::InvalidLiteral,
                        "integer literal out of range".to_string(),
                    )
                })?;
                Ok(Expr::Nat {
                    value,
                    span: t.span,
                })
            }
            TokenKind::Float => {
                let t = self.advance();
                let text = self.slice(t.span);
                let value = text.parse::<f64>().map_err(|_| {
                    SyntaxError::new(
                        t.span,
                        ErrorKind::InvalidLiteral,
                        "malformed float literal".to_string(),
                    )
                })?;
                Ok(Expr::Float {
                    value,
                    span: t.span,
                })
            }
            TokenKind::Str => {
                let t = self.advance();
                let value = unescape(self.slice(t.span))
                    .map_err(|msg| SyntaxError::new(t.span, ErrorKind::InvalidString, msg))?;
                Ok(Expr::Str {
                    value,
                    span: t.span,
                })
            }
            TokenKind::LParen => {
                let open = self.advance();
                let inner = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                let span = Span::covering(open.span, self.prev_span);
                Ok(Expr::Paren {
                    inner: Box::new(inner),
                    span,
                })
            }
            TokenKind::Minus => {
                let minus = self.advance();
                let operand = self.parse_pow()?;
                let span = Span::covering(minus.span, self.prev_span);
                Ok(Expr::UnOp {
                    op: UnOp::Neg,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::Lambda => self.parse_lambda(),
            _ => {
                let t = self.peek_token();
                if t.kind == TokenKind::Eof {
                    Err(SyntaxError::new(
                        t.span,
                        ErrorKind::UnexpectedEof,
                        "unexpected end of input; expected an expression".to_string(),
                    ))
                } else {
                    Err(SyntaxError::new(
                        t.span,
                        ErrorKind::UnexpectedToken,
                        format!("expected an expression, found {actual:?}", actual = t.kind),
                    ))
                }
            }
        }
    }

    /// `λ (x : A) (y : B). body`, with at least one annotated parameter.
    ///
    /// The body extends as far right as possible (it is parsed with
    /// [`parse_expr`](Parser::parse_expr)), exactly like the arrow
    /// operator: `λ (x : Nat). x + 1` is `Lam([x], x + 1)`, and in
    /// `f(λ (x : Nat). x, y)` the `,` terminates the lambda.
    fn parse_lambda(&mut self) -> Result<Expr, SyntaxError> {
        let lam = self.advance();
        if self.peek() != TokenKind::LParen {
            return Err(self.err_here(
                ErrorKind::UnexpectedToken,
                "expected `(` with an annotated parameter after `λ`".to_string(),
            ));
        }
        let mut params = Vec::new();
        loop {
            self.expect(TokenKind::LParen)?;
            let (name, name_span) = self.expect_binding_ident("a parameter name")?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            let span = Span::covering(name_span, self.prev_span);
            params.push(LamParam {
                name,
                name_span,
                ty,
                span,
            });
            if self.peek() != TokenKind::LParen {
                break;
            }
        }
        self.expect(TokenKind::Dot)?;
        let body = self.parse_expr()?;
        let span = Span::covering(lam.span, self.prev_span);
        Ok(Expr::Lam {
            params,
            body: Box::new(body),
            span,
        })
    }
}

/// The span of an expression node.
fn expr_span(e: &Expr) -> Span {
    match e {
        Expr::Ident { span, .. }
        | Expr::Nat { span, .. }
        | Expr::Float { span, .. }
        | Expr::Str { span, .. }
        | Expr::Field { span, .. }
        | Expr::Index { span, .. }
        | Expr::Call { span, .. }
        | Expr::BinOp { span, .. }
        | Expr::UnOp { span, .. }
        | Expr::Paren { span, .. }
        | Expr::Lam { span, .. } => *span,
        Expr::If(e) => e.span,
        Expr::Match(e) => e.span,
    }
}

/// Unescape a string literal *including* its surrounding quotes.
fn unescape(text: &str) -> Result<String, String> {
    let body = text
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| "malformed string literal".to_string())?;
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(other) => {
                    return Err(format!("unknown string escape `\\{other}`"));
                }
                None => return Err("unterminated escape".to_string()),
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Result<Module, Vec<SyntaxError>> {
        parse_module(source)
    }

    fn parse_one_fn(source: &str) -> FunctionDef {
        let m = parse(source).expect("parse should succeed");
        assert_eq!(m.functions.len(), 1);
        m.functions.into_iter().next().unwrap()
    }

    // ---- Module ------------------------------------------------------

    #[test]
    fn module_decl_with_multi_segment_path() {
        let m = parse("Module Standard.Geometry.Shapes\n").unwrap();
        let decl = m.decl.expect("module decl");
        let segs: Vec<&str> = decl.path.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(segs, vec!["Standard", "Geometry", "Shapes"]);
    }

    #[test]
    fn empty_file_parses_to_empty_module() {
        let m = parse("").unwrap();
        assert!(m.decl.is_none());
        assert!(m.imports.is_empty());
        assert!(m.functions.is_empty());
    }

    // ---- Imports -----------------------------------------------------

    #[test]
    fn import_without_names() {
        let m = parse("Import Standard.Nat\n").unwrap();
        assert_eq!(m.imports.len(), 1);
        assert!(m.imports[0].names.is_none());
    }

    #[test]
    fn import_with_name_list() {
        let m = parse("Import Standard.Bool (true, false)\n").unwrap();
        assert_eq!(m.imports.len(), 1);
        let names = m.imports[0].names.as_ref().expect("names");
        let texts: Vec<&str> = names.iter().map(|n| n.text.as_str()).collect();
        assert_eq!(texts, vec!["true", "false"]);
    }

    // ---- Functions ---------------------------------------------------

    #[test]
    fn function_zero_one_three_params() {
        let f = parse_one_fn("Function f():\n    return 1\n");
        assert!(f.params.is_empty());
        assert!(f.return_ty.is_none());

        let f = parse_one_fn("Function f(x: Nat):\n    return x\n");
        assert_eq!(f.params.len(), 1);
        assert_eq!(f.params[0].name, "x");

        let f = parse_one_fn("Function f(a: Nat, b: Nat, c: Nat):\n    return a\n");
        assert_eq!(f.params.len(), 3);
    }

    #[test]
    fn function_with_and_without_return_type() {
        let f = parse_one_fn("Function f(x: Nat) -> Nat:\n    return x\n");
        assert!(f.return_ty.is_some());

        let f = parse_one_fn("Function f(x: Nat):\n    return x\n");
        assert!(f.return_ty.is_none());
    }

    // ---- Expressions -------------------------------------------------

    fn parse_expr_only(source: &str) -> Expr {
        // Wrap in a return statement to reuse the full pipeline.
        let src = format!("Function f():\n    return {source}\n");
        let f = parse_one_fn(&src);
        assert_eq!(f.body.len(), 1);
        match f.body.into_iter().next().unwrap() {
            Stmt::Return { value, .. } => value,
            Stmt::Let { .. } => panic!("expected return"),
        }
    }

    #[test]
    fn add_mul_precedence() {
        // 1 + 2 * 3 == BinOp(1, Add, BinOp(2, Mul, 3))
        match parse_expr_only("1 + 2 * 3") {
            Expr::BinOp {
                op: BinOp::Add,
                rhs,
                ..
            } => match *rhs {
                Expr::BinOp { op: BinOp::Mul, .. } => {}
                other => panic!("expected Mul on rhs, got {other:?}"),
            },
            other => panic!("expected Add on top, got {other:?}"),
        }
    }

    #[test]
    fn pow_is_right_associative() {
        // 2 ^ 3 ^ 4 == 2 ^ (3 ^ 4)
        match parse_expr_only("2 ^ 3 ^ 4") {
            Expr::BinOp {
                op: BinOp::Pow,
                rhs,
                ..
            } => match *rhs {
                Expr::BinOp { op: BinOp::Pow, .. } => {}
                other => panic!("expected right-nested Pow, got {other:?}"),
            },
            other => panic!("expected Pow on top, got {other:?}"),
        }
    }

    #[test]
    fn unary_minus_binds_looser_than_pow() {
        // -x ^ 2 == Neg(Pow(x, 2))
        match parse_expr_only("-x ^ 2") {
            Expr::UnOp {
                op: UnOp::Neg,
                operand,
                ..
            } => match *operand {
                Expr::BinOp { op: BinOp::Pow, .. } => {}
                other => panic!("expected Pow under Neg, got {other:?}"),
            },
            other => panic!("expected Neg on top, got {other:?}"),
        }
    }

    #[test]
    fn field_index_call_chain_left_to_right() {
        // a.b[0](c) == Call(Index(Field(a, b), 0), [c])
        match parse_expr_only("a.b[0](c)") {
            Expr::Call { func, args, .. } => {
                assert_eq!(args.len(), 1);
                match *func {
                    Expr::Index { .. } => {}
                    other => panic!("expected Index as callee, got {other:?}"),
                }
            }
            other => panic!("expected Call on top, got {other:?}"),
        }
    }

    #[test]
    fn chained_comparison_is_an_error() {
        assert!(parse_module("Function f():\n    return a < b < c\n").is_err());
    }

    // ---- Errors ------------------------------------------------------

    #[test]
    fn unknown_top_level_form_is_not_implemented() {
        let errs = parse_module("Foobar foo:\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ErrorKind::NotImplemented);
    }

    // ---- Lambda --------------------------------------------------------

    #[test]
    fn lambda_single_param_parses() {
        match parse_expr_only("λ (x : Nat). x") {
            Expr::Lam { params, body, .. } => {
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "x");
                assert!(matches!(*body, Expr::Ident { .. }));
            }
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn lambda_two_params_parse() {
        match parse_expr_only("λ (x : Nat) (y : Bool). x") {
            Expr::Lam { params, .. } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name, "x");
                assert_eq!(params[1].name, "y");
            }
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn lambda_body_extends_right() {
        // λ (x : Nat). x + 1  ==  Lam([x], x + 1)
        match parse_expr_only("λ (x : Nat). x + 1") {
            Expr::Lam { body, .. } => assert!(matches!(*body, Expr::BinOp { .. })),
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn lambda_missing_annotation_is_an_error() {
        assert!(parse_module("Function f():\n    return λ x . x\n").is_err());
    }

    #[test]
    fn lambda_without_params_is_an_error() {
        assert!(parse_module("Function f():\n    return λ . x\n").is_err());
    }

    // ---- If expressions ------------------------------------------------

    fn parse_if_only(source: &str) -> IfExpr {
        match parse_expr_only(source) {
            Expr::If(e) => e,
            other => panic!("expected If, got {other:?}"),
        }
    }

    fn ident_text(e: &Expr) -> &str {
        match e {
            Expr::Ident { text, .. } => text,
            other => panic!("expected Ident, got {other:?}"),
        }
    }

    #[test]
    fn if_parses_branches() {
        let e = parse_if_only("if a then b else c");
        assert_eq!(ident_text(&e.cond), "a");
        assert_eq!(ident_text(&e.then_branch), "b");
        assert_eq!(ident_text(&e.else_branch), "c");
    }

    #[test]
    fn if_condition_extends_right() {
        match parse_if_only("if a + 1 then b else c").cond.as_ref() {
            Expr::BinOp { op: BinOp::Add, .. } => {}
            other => panic!("expected Add condition, got {other:?}"),
        }
    }

    #[test]
    fn if_then_branch_extends_right() {
        match parse_if_only("if a then b + 1 else c").then_branch.as_ref() {
            Expr::BinOp { op: BinOp::Add, .. } => {}
            other => panic!("expected Add then-branch, got {other:?}"),
        }
    }

    #[test]
    fn if_else_branch_extends_right() {
        match parse_if_only("if a then b else c + 1").else_branch.as_ref() {
            Expr::BinOp { op: BinOp::Add, .. } => {}
            other => panic!("expected Add else-branch, got {other:?}"),
        }
    }

    #[test]
    fn if_nests_in_then_branch() {
        match parse_if_only("if a then if b then x else y else z")
            .then_branch
            .as_ref()
        {
            Expr::If(_) => {}
            other => panic!("expected nested If, got {other:?}"),
        }
    }

    #[test]
    fn if_nests_in_else_branch() {
        match parse_if_only("if a then x else if b then y else z")
            .else_branch
            .as_ref()
        {
            Expr::If(_) => {}
            other => panic!("expected nested If, got {other:?}"),
        }
    }

    #[test]
    fn if_missing_then_is_an_error() {
        assert!(parse_module("Function f():\n    return if a\n").is_err());
    }

    #[test]
    fn if_missing_else_is_an_error() {
        assert!(parse_module("Function f():\n    return if a then b\n").is_err());
    }

    #[test]
    fn if_missing_condition_is_an_error() {
        assert!(parse_module("Function f():\n    return if then b else c\n").is_err());
    }

    // ---- Let statements ------------------------------------------------

    fn parse_body_stmts(source: &str) -> Vec<Stmt> {
        let m = parse_module(source).expect("parse should succeed");
        assert_eq!(m.functions.len(), 1);
        m.functions.into_iter().next().unwrap().body
    }

    #[test]
    fn let_then_return_parses_to_two_statements() {
        let body = parse_body_stmts("Function f() -> Nat:\n    let x : Nat = 0\n    return x\n");
        assert_eq!(body.len(), 2);
        assert!(matches!(body[0], Stmt::Let { .. }));
        assert!(matches!(body[1], Stmt::Return { .. }));
    }

    #[test]
    fn let_without_return_is_an_error() {
        assert!(parse_module("Function f() -> Nat:\n    let x : Nat = 0\n").is_err());
    }

    #[test]
    fn bare_return_is_a_valid_body() {
        let body = parse_body_stmts("Function f() -> Nat:\n    return 0\n");
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], Stmt::Return { .. }));
    }

    #[test]
    fn let_missing_annotation_is_an_error() {
        assert!(parse_module("Function f() -> Nat:\n    let x = 0\n    return x\n").is_err());
    }

    #[test]
    fn let_missing_eq_is_an_error() {
        assert!(parse_module("Function f() -> Nat:\n    let x : Nat 0\n    return x\n").is_err());
    }

    #[test]
    fn two_lets_then_return_parse_in_order() {
        let body = parse_body_stmts(
            "Function f() -> Nat:\n    let x : Nat = 0\n    let y : Nat = x\n    return y\n",
        );
        assert_eq!(body.len(), 3);
        match &body[0] {
            Stmt::Let { name, .. } => assert_eq!(name, "x"),
            other => panic!("expected Let, got {other:?}"),
        }
        match &body[1] {
            Stmt::Let { name, .. } => assert_eq!(name, "y"),
            other => panic!("expected Let, got {other:?}"),
        }
        assert!(matches!(body[2], Stmt::Return { .. }));
    }

    // ---- Reserved binding names ----------------------------------------

    #[test]
    fn keyword_as_function_name_is_an_error() {
        assert!(parse_module("Function if() -> Nat:\n    return 0\n").is_err());
    }

    #[test]
    fn keyword_as_let_name_is_an_error() {
        assert!(
            parse_module("Function f() -> Nat:\n    let return : Nat = 0\n    return 0\n").is_err()
        );
    }

    #[test]
    fn match_cannot_be_a_function_name() {
        assert!(parse_module("Function match() -> Nat:\n    return 0\n").is_err());
    }

    #[test]
    fn case_cannot_be_a_parameter_name() {
        assert!(parse_module("Function f(case : Nat) -> Nat:\n    return case\n").is_err());
    }

    #[test]
    fn module_cannot_be_a_let_binding_name() {
        assert!(
            parse_module("Function f() -> Nat:\n    let Module : Nat = 0\n    return 0\n").is_err()
        );
    }

    #[test]
    fn keyword_as_match_name_is_an_error() {
        assert!(parse_module("Function match() -> Nat:\n    return 0\n").is_err());
    }

    #[test]
    fn keyword_as_parameter_name_is_an_error() {
        assert!(parse_module("Function f(case : Nat) -> Nat:\n    return case\n").is_err());
    }

    #[test]
    fn keyword_as_module_name_is_an_error() {
        assert!(parse_module("Function Module() -> Nat:\n    return 0\n").is_err());
    }

    // ---- Match expressions ---------------------------------------------

    /// Parse `return <match>` inside a function and return the match node.
    fn parse_match_only(arms: &str) -> MatchExpr {
        let src = format!("Function f(n : Nat) -> Nat:\n    return match n:\n{arms}\n");
        let m = parse_module(&src).expect("parse should succeed");
        assert_eq!(m.functions.len(), 1);
        let body = &m.functions[0].body;
        assert_eq!(body.len(), 1);
        match &body[0] {
            Stmt::Return { value, .. } => match value {
                Expr::Match(e) => e.clone(),
                other => panic!("expected Match, got {other:?}"),
            },
            other => panic!("expected Return, got {other:?}"),
        }
    }

    #[test]
    fn match_two_arms_parse() {
        let m = parse_match_only("        case Nat.zero => Nat.zero\n        case Nat.succ k => k");
        assert_eq!(m.arms.len(), 2);
    }

    #[test]
    fn match_patterns_resolve_to_nodes() {
        let m = parse_match_only("        case Nat.zero => Nat.zero\n        case Nat.succ k => k");
        match &m.arms[0].pattern {
            Pattern::Constructor { name, args, .. } => {
                assert_eq!(name, "Nat.zero");
                assert!(args.is_empty());
            }
            other => panic!("expected Constructor, got {other:?}"),
        }
        match &m.arms[1].pattern {
            Pattern::Constructor { name, args, .. } => {
                assert_eq!(name, "Nat.succ");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Constructor, got {other:?}"),
        }
    }

    #[test]
    fn wildcard_arm_parses() {
        let m = parse_match_only("        case _ => Nat.zero");
        assert_eq!(m.arms.len(), 1);
        assert!(matches!(m.arms[0].pattern, Pattern::Wildcard(_)));
    }

    #[test]
    fn variable_arm_parses() {
        let m = parse_match_only("        case k => k");
        assert_eq!(m.arms.len(), 1);
        match &m.arms[0].pattern {
            Pattern::Var { name, .. } => assert_eq!(name, "k"),
            other => panic!("expected Var, got {other:?}"),
        }
    }

    #[test]
    fn nullary_constructor_parses() {
        let m = parse_match_only("        case Nat.zero => Nat.zero");
        match &m.arms[0].pattern {
            Pattern::Constructor { args, .. } => assert!(args.is_empty()),
            other => panic!("expected Constructor, got {other:?}"),
        }
    }

    #[test]
    fn constructor_arg_parses() {
        let m = parse_match_only("        case Nat.succ(k) => k");
        match &m.arms[0].pattern {
            Pattern::Constructor { args, .. } => match &args[..] {
                [PatternArg::Var { name, .. }] => assert_eq!(name, "k"),
                other => panic!("expected one Var arg, got {other:?}"),
            },
            other => panic!("expected Constructor, got {other:?}"),
        }
    }

    #[test]
    fn multi_arg_constructor_parses_name_agnostically() {
        let m = parse_match_only("        case Pair(fst, snd) => fst");
        match &m.arms[0].pattern {
            Pattern::Constructor { name, args, .. } => {
                assert_eq!(name, "Pair");
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected Constructor, got {other:?}"),
        }
    }

    #[test]
    fn nested_wildcard_arg_parses() {
        let m = parse_match_only("        case Nat.succ(_) => Nat.zero");
        match &m.arms[0].pattern {
            Pattern::Constructor { args, .. } => match &args[..] {
                [PatternArg::Wildcard(_)] => {}
                other => panic!("expected one Wildcard arg, got {other:?}"),
            },
            other => panic!("expected Constructor, got {other:?}"),
        }
    }

    #[test]
    fn compound_scrutinee_parses() {
        let src = "Function f(n : Nat) -> Nat:\n    return match n + 1:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n";
        let m2 = parse_module(src).expect("parse should succeed");
        let body = &m2.functions[0].body;
        match &body[0] {
            Stmt::Return {
                value: Expr::Match(e),
                ..
            } => match e.scrutinee.as_ref() {
                Expr::BinOp { op: BinOp::Add, .. } => {}
                other => panic!("expected Add scrutinee, got {other:?}"),
            },
            other => panic!("expected Return(Match), got {other:?}"),
        }
    }

    #[test]
    fn if_scrutinee_parses() {
        let src = "Function f(c : Bool, a : Nat, b : Nat) -> Nat:\n    return match if c then a else b:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n";
        let m = parse_module(src).expect("parse should succeed");
        let body = &m.functions[0].body;
        match &body[0] {
            Stmt::Return {
                value: Expr::Match(e),
                ..
            } => match e.scrutinee.as_ref() {
                Expr::If(_) => {}
                other => panic!("expected If scrutinee, got {other:?}"),
            },
            other => panic!("expected Return(Match), got {other:?}"),
        }
    }

    #[test]
    fn match_missing_colon_is_an_error() {
        assert!(parse_module("Function f(n : Nat) -> Nat:\n    return match n\n").is_err());
    }

    #[test]
    fn match_no_arms_is_an_error() {
        // `match n:` followed by a dedent (no indented arms).
        assert!(
            parse_module("Function f(n : Nat) -> Nat:\n    return match n:\nFunction g() -> Nat:\n    return 0\n")
                .is_err()
        );
    }

    #[test]
    fn match_missing_case_is_an_error() {
        assert!(
            parse_module(
                "Function f(n : Nat) -> Nat:\n    return match n:\n        Nat.zero => Nat.zero\n"
            )
            .is_err()
        );
    }

    #[test]
    fn match_missing_fatarrow_is_an_error() {
        assert!(
            parse_module(
                "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero Nat.zero\n"
            )
            .is_err()
        );
    }

    #[test]
    fn nested_constructor_pattern_rejected() {
        let errs =
            parse_module("Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.succ(Nat.zero) => Nat.zero\n")
                .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ErrorKind::NotImplemented);
    }

    #[test]
    fn mixed_arm_indentation_is_an_error() {
        assert!(
            parse_module(
                "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n      case Nat.succ k => k\n"
            )
            .is_err()
        );
    }

    // ---- Structure declarations ----------------------------------------

    fn parse_structures_only(src: &str) -> Module {
        parse_module(src).expect("parse should succeed")
    }

    #[test]
    fn structure_two_nat_fields_parse() {
        let m = parse_structures_only("Structure Point2D:\n    x : Nat\n    y : Nat\n");
        assert_eq!(m.structures.len(), 1);
        let s = &m.structures[0];
        assert_eq!(s.name, "Point2D");
        assert!(s.params.is_empty());
        assert_eq!(s.fields.len(), 2);
        assert_eq!(s.fields[0].name, "x");
        assert_eq!(s.fields[1].name, "y");
    }

    #[test]
    fn structure_with_parameter_parses() {
        let m = parse_structures_only("Structure Box(A : Type):\n    contents : A\n");
        assert_eq!(m.structures.len(), 1);
        let s = &m.structures[0];
        assert_eq!(s.name, "Box");
        assert_eq!(s.params.len(), 1);
        assert_eq!(s.params[0].name, "A");
        assert_eq!(s.fields.len(), 1);
        assert_eq!(s.fields[0].name, "contents");
    }

    #[test]
    fn structure_recursive_field_parses() {
        let m = parse_structures_only("Structure Wrapper:\n    value : Nat\n    inner : Wrapper\n");
        assert_eq!(m.structures.len(), 1);
        assert_eq!(m.structures[0].fields.len(), 2);
    }

    #[test]
    fn structure_field_compound_type_parses() {
        let m = parse_structures_only("Structure Endo:\n    f : Nat -> Nat\n");
        assert_eq!(m.structures.len(), 1);
        assert_eq!(m.structures[0].fields.len(), 1);
    }

    #[test]
    fn structure_field_referencing_parameter_parses() {
        let m = parse_structures_only(
            "Structure Stream(A : Type):\n    head : A\n    tail : Stream(A)\n",
        );
        let s = &m.structures[0];
        assert_eq!(s.params.len(), 1);
        assert_eq!(s.fields.len(), 2);
    }

    #[test]
    fn multiple_structures_parse() {
        let m = parse_structures_only(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nStructure Wrapper:\n    value : Nat\n    inner : Wrapper\n",
        );
        assert_eq!(m.structures.len(), 2);
        assert_eq!(m.structures[0].name, "Point2D");
        assert_eq!(m.structures[1].name, "Wrapper");
    }

    #[test]
    fn structure_followed_by_function_parses() {
        let m = parse_structures_only(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nFunction origin() -> Point2D:\n    return Point2D.mk(0, 0)\n",
        );
        assert_eq!(m.structures.len(), 1);
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "origin");
    }

    #[test]
    fn structure_with_no_fields_is_an_error() {
        assert!(parse_module("Structure Foo:\n").is_err());
    }

    #[test]
    fn structure_missing_name_is_an_error() {
        assert!(parse_module("Structure :\n    x : Nat\n").is_err());
    }

    #[test]
    fn structure_missing_colon_is_an_error() {
        assert!(parse_module("Structure Foo\n    x : Nat\n").is_err());
    }

    #[test]
    fn structure_field_missing_colon_is_an_error() {
        assert!(parse_module("Structure Foo:\n    x = Nat\n").is_err());
    }

    #[test]
    fn structure_name_is_reserved() {
        assert!(parse_module("Function Structure() -> Nat:\n    return 0\n").is_err());
    }

    // ---- Theorems ---------------------------------------------------

    fn parse_theorem_only(source: &str) -> TheoremDef {
        let m = parse_module(source).expect("parse should succeed");
        assert_eq!(m.theorems.len(), 1);
        m.theorems.into_iter().next().unwrap()
    }

    #[test]
    fn theorem_no_given_no_assume_parses() {
        let t = parse_theorem_only(
            "Theorem zero_eq_zero:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        );
        assert_eq!(t.name, "zero_eq_zero");
        assert!(t.given.is_empty());
        assert!(t.assume.is_empty());
    }

    #[test]
    fn theorem_with_given_parses() {
        let t = parse_theorem_only(
            "Theorem refl_nat:\n    Given:\n        n : Nat\n    Show: Eq(Nat, n, n)\n    Proof: Eq.refl(Nat, n)\n    QED\n",
        );
        assert_eq!(t.given.len(), 1);
        assert_eq!(t.given[0].name, "n");
    }

    #[test]
    fn theorem_with_two_given_parses() {
        let t = parse_theorem_only(
            "Theorem f:\n    Given:\n        a : Nat\n        b : Nat\n    Show: Eq(Nat, a, a)\n    Proof: Eq.refl(Nat, a)\n    QED\n",
        );
        assert_eq!(t.given.len(), 2);
        assert_eq!(t.given[0].name, "a");
        assert_eq!(t.given[1].name, "b");
    }

    #[test]
    fn theorem_with_assume_parses() {
        let t = parse_theorem_only(
            "Theorem modus_ponens:\n    Assume:\n        h : P\n    Show: P\n    Proof: h\n    QED\n",
        );
        assert_eq!(t.assume.len(), 1);
        assert_eq!(t.assume[0].name, "h");
    }

    #[test]
    fn theorem_with_given_and_assume_parses() {
        let t = parse_theorem_only(
            "Theorem f:\n    Given:\n        n : Nat\n    Assume:\n        h : Eq(Nat, n, 0)\n    Show: Eq(Nat, n, 0)\n    Proof: h\n    QED\n",
        );
        assert_eq!(t.given.len(), 1);
        assert_eq!(t.assume.len(), 1);
    }

    #[test]
    fn theorem_missing_show_is_an_error() {
        assert!(parse_module("Theorem f:\n    Proof: x\n    QED\n").is_err());
    }

    #[test]
    fn theorem_missing_proof_is_an_error() {
        assert!(parse_module("Theorem f:\n    Show: P\n    QED\n").is_err());
    }

    #[test]
    fn theorem_missing_qed_is_an_error() {
        assert!(parse_module("Theorem f:\n    Show: P\n    Proof: x\n").is_err());
    }

    #[test]
    fn theorem_and_function_coexist() {
        let m = parse_module(
            "Function f() -> Nat:\n    return 0\n\nTheorem t:\n    Show: P\n    Proof: h\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.theorems.len(), 1);
    }

    // ---- Declarative proof steps -----------------------------------

    fn parse_theorem_only_with_steps(source: &str) -> TheoremDef {
        let m = parse_module(source).expect("parse should succeed");
        assert_eq!(m.theorems.len(), 1);
        m.theorems.into_iter().next().unwrap()
    }

    fn steps_of(t: &TheoremDef) -> &[ProofStep] {
        match &t.proof {
            ProofBlock::Steps(s) => s,
            ProofBlock::Term(_) => panic!("expected step block, got term"),
        }
    }

    #[test]
    fn term_mode_proof_still_parses() {
        let t = parse_theorem_only_with_steps("Theorem t:\n    Show: P\n    Proof: h\n    QED\n");
        assert!(matches!(t.proof, ProofBlock::Term(_)));
    }

    #[test]
    fn single_exact_step_parses() {
        let t = parse_theorem_only_with_steps(
            "Theorem t:\n    Show: P\n    Proof:\n        1. Exact h\n    QED\n",
        );
        let s = steps_of(&t);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].number, 1);
        assert!(matches!(s[0].kind, ProofStepKind::Exact { .. }));
    }

    #[test]
    fn let_have_exact_chain_parses() {
        let t = parse_theorem_only_with_steps(
            "Theorem t:\n    Show: P\n    Proof:\n        1. Let n : Nat = 0\n        2. Have h : P = h0\n        3. Exact h\n    QED\n",
        );
        let s = steps_of(&t);
        assert_eq!(s.len(), 3);
        assert!(matches!(s[0].kind, ProofStepKind::Let { .. }));
        assert!(matches!(s[1].kind, ProofStepKind::Have { .. }));
        assert!(matches!(s[2].kind, ProofStepKind::Exact { .. }));
        assert_eq!(s[0].number, 1);
        assert_eq!(s[1].number, 2);
        assert_eq!(s[2].number, 3);
    }

    #[test]
    fn let_without_annotation_parses() {
        let t = parse_theorem_only_with_steps(
            "Theorem t:\n    Show: P\n    Proof:\n        1. Let n = 0\n        2. Exact h\n    QED\n",
        );
        match &steps_of(&t)[0].kind {
            ProofStepKind::Let { ty, .. } => assert!(ty.is_none()),
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn step_justification_parses() {
        let t = parse_theorem_only_with_steps(
            "Theorem t:\n    Show: P\n    Proof:\n        1. Exact h [by refl]\n    QED\n",
        );
        let s = steps_of(&t);
        assert_eq!(s[0].justification.as_deref(), Some("by refl"));
    }

    #[test]
    fn non_sequential_step_number_rejected() {
        let errs = parse_module(
            "Theorem t:\n    Show: P\n    Proof:\n        1. Exact h\n        3. Exact h\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("expected step number 2"));
    }

    #[test]
    fn step_number_not_starting_at_one_rejected() {
        let errs =
            parse_module("Theorem t:\n    Show: P\n    Proof:\n        2. Exact h\n    QED\n")
                .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("expected step number 1"));
    }

    #[test]
    fn empty_proof_block_rejected() {
        let errs = parse_module("Theorem t:\n    Show: P\n    Proof:\n    QED\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(
            errs[0].message.contains("at least one step")
                || errs[0].message.contains("expected an indented block")
        );
    }

    #[test]
    fn unknown_step_keyword_rejected() {
        let errs =
            parse_module("Theorem t:\n    Show: P\n    Proof:\n        1. Assume q\n    QED\n")
                .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(
            errs[0]
                .message
                .contains("expected `Let`, `Have`, `Exact`, or `From`"),
            "stderr was:\n{}",
            errs[0].message
        );
    }

    #[test]
    fn from_step_parses() {
        let t = parse_theorem_only_with_steps(
            "Theorem t:\n    Assume:\n        h : P\n    Show: P\n    Proof:\n        1. From h\n    QED\n",
        );
        match &steps_of(&t)[0].kind {
            ProofStepKind::From { name, .. } => assert_eq!(name, "h"),
            other => panic!("expected From, got {other:?}"),
        }
    }

    #[test]
    fn from_step_with_justification_parses() {
        let t = parse_theorem_only_with_steps(
            "Theorem t:\n    Assume:\n        h : P\n    Show: P\n    Proof:\n        1. From h [by assumption]\n    QED\n",
        );
        let s = steps_of(&t);
        assert!(matches!(s[0].kind, ProofStepKind::From { .. }));
        assert_eq!(s[0].justification.as_deref(), Some("by assumption"));
    }

    #[test]
    fn from_step_missing_name_rejected() {
        let errs = parse_module("Theorem t:\n    Show: P\n    Proof:\n        1. From\n    QED\n")
            .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("identifier"));
    }

    #[test]
    fn unterminated_justification_rejected() {
        let errs = parse_module(
            "Theorem t:\n    Show: P\n    Proof:\n        1. Exact h [by refl\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("unterminated") || errs[0].message.contains("expected"),);
    }
}
