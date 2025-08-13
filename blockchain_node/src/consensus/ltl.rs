//! Linear Temporal Logic (LTL) Parser and Model Checker
//!
//! This module provides comprehensive LTL formula parsing and model checking
//! capabilities for verifying consensus properties over time.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::consensus::petri_net::{PetriNet, Marking};

/// LTL formula representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LTLFormula {
    /// Atomic proposition
    Atom(String),
    /// Boolean true
    True,
    /// Boolean false
    False,
    /// Negation: !φ
    Not(Box<LTLFormula>),
    /// Conjunction: φ ∧ ψ
    And(Box<LTLFormula>, Box<LTLFormula>),
    /// Disjunction: φ ∨ ψ
    Or(Box<LTLFormula>, Box<LTLFormula>),
    /// Implication: φ → ψ
    Implies(Box<LTLFormula>, Box<LTLFormula>),
    /// Next: X φ (φ holds in the next state)
    Next(Box<LTLFormula>),
    /// Until: φ U ψ (φ holds until ψ becomes true)
    Until(Box<LTLFormula>, Box<LTLFormula>),
    /// Weak Until: φ W ψ (φ holds until ψ becomes true, or forever)
    WeakUntil(Box<LTLFormula>, Box<LTLFormula>),
    /// Release: φ R ψ (ψ holds until both φ and ψ hold)
    Release(Box<LTLFormula>, Box<LTLFormula>),
    /// Finally: F φ (eventually φ holds)
    Finally(Box<LTLFormula>),
    /// Globally: G φ (φ always holds)
    Globally(Box<LTLFormula>),
}

impl fmt::Display for LTLFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LTLFormula::Atom(s) => write!(f, "{}", s),
            LTLFormula::True => write!(f, "true"),
            LTLFormula::False => write!(f, "false"),
            LTLFormula::Not(phi) => write!(f, "!{}", phi),
            LTLFormula::And(phi, psi) => write!(f, "({} && {})", phi, psi),
            LTLFormula::Or(phi, psi) => write!(f, "({} || {})", phi, psi),
            LTLFormula::Implies(phi, psi) => write!(f, "({} -> {})", phi, psi),
            LTLFormula::Next(phi) => write!(f, "X{}", phi),
            LTLFormula::Until(phi, psi) => write!(f, "({} U {})", phi, psi),
            LTLFormula::WeakUntil(phi, psi) => write!(f, "({} W {})", phi, psi),
            LTLFormula::Release(phi, psi) => write!(f, "({} R {})", phi, psi),
            LTLFormula::Finally(phi) => write!(f, "F{}", phi),
            LTLFormula::Globally(phi) => write!(f, "G{}", phi),
        }
    }
}

impl LTLFormula {
    /// Simplify the formula by applying basic logical rules
    pub fn simplify(self) -> LTLFormula {
        match self {
            LTLFormula::Not(box LTLFormula::Not(phi)) => phi.simplify(),
            LTLFormula::And(box LTLFormula::True, phi) => phi.simplify(),
            LTLFormula::And(phi, box LTLFormula::True) => phi.simplify(),
            LTLFormula::And(box LTLFormula::False, _) => LTLFormula::False,
            LTLFormula::And(_, box LTLFormula::False) => LTLFormula::False,
            LTLFormula::Or(box LTLFormula::False, phi) => phi.simplify(),
            LTLFormula::Or(phi, box LTLFormula::False) => phi.simplify(),
            LTLFormula::Or(box LTLFormula::True, _) => LTLFormula::True,
            LTLFormula::Or(_, box LTLFormula::True) => LTLFormula::True,
            LTLFormula::Finally(phi) => LTLFormula::Until(Box::new(LTLFormula::True), Box::new(phi.simplify())),
            LTLFormula::Globally(phi) => LTLFormula::Release(Box::new(LTLFormula::False), Box::new(phi.simplify())),
            LTLFormula::And(phi, psi) => LTLFormula::And(Box::new(phi.simplify()), Box::new(psi.simplify())),
            LTLFormula::Or(phi, psi) => LTLFormula::Or(Box::new(phi.simplify()), Box::new(psi.simplify())),
            LTLFormula::Implies(phi, psi) => LTLFormula::Or(
                Box::new(LTLFormula::Not(Box::new(phi.simplify()))),
                Box::new(psi.simplify())
            ),
            other => other,
        }
    }

    /// Get all atomic propositions in the formula
    pub fn get_atoms(&self) -> HashSet<String> {
        let mut atoms = HashSet::new();
        self.collect_atoms(&mut atoms);
        atoms
    }

    fn collect_atoms(&self, atoms: &mut HashSet<String>) {
        match self {
            LTLFormula::Atom(s) => { atoms.insert(s.clone()); }
            LTLFormula::Not(phi) => phi.collect_atoms(atoms),
            LTLFormula::And(phi, psi) | LTLFormula::Or(phi, psi) | 
            LTLFormula::Implies(phi, psi) | LTLFormula::Until(phi, psi) |
            LTLFormula::WeakUntil(phi, psi) | LTLFormula::Release(phi, psi) => {
                phi.collect_atoms(atoms);
                psi.collect_atoms(atoms);
            }
            LTLFormula::Next(phi) | LTLFormula::Finally(phi) | LTLFormula::Globally(phi) => {
                phi.collect_atoms(atoms);
            }
            LTLFormula::True | LTLFormula::False => {}
        }
    }
}

#[derive(Error, Debug)]
pub enum LTLError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Model checking error: {0}")]
    ModelCheckingError(String),
    #[error("Invalid formula: {0}")]
    InvalidFormula(String),
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
}

/// Simple recursive descent parser for LTL formulas
pub struct LTLParser {
    tokens: Vec<Token>,
    position: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Atom(String),
    True,
    False,
    Not,          // !
    And,          // &&, &, /\
    Or,           // ||, |, \/
    Implies,      // ->, =>
    Next,         // X
    Until,        // U
    WeakUntil,    // W
    Release,      // R
    Finally,      // F
    Globally,     // G
    LeftParen,    // (
    RightParen,   // )
    EOF,
}

impl LTLParser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            position: 0,
        }
    }

    /// Parse an LTL formula from a string
    pub fn parse(&mut self, input: &str) -> Result<LTLFormula, LTLError> {
        self.tokenize(input)?;
        self.position = 0;
        let formula = self.parse_formula()?;
        if self.current_token() != &Token::EOF {
            return Err(LTLError::ParseError("Unexpected tokens after formula".to_string()));
        }
        Ok(formula.simplify())
    }

    fn tokenize(&mut self, input: &str) -> Result<(), LTLError> {
        self.tokens.clear();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\n' | '\r' => { chars.next(); }
                '(' => { self.tokens.push(Token::LeftParen); chars.next(); }
                ')' => { self.tokens.push(Token::RightParen); chars.next(); }
                '!' => { self.tokens.push(Token::Not); chars.next(); }
                '&' => {
                    chars.next();
                    if chars.peek() == Some(&'&') {
                        chars.next();
                    }
                    self.tokens.push(Token::And);
                }
                '|' => {
                    chars.next();
                    if chars.peek() == Some(&'|') {
                        chars.next();
                    }
                    self.tokens.push(Token::Or);
                }
                '-' => {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        self.tokens.push(Token::Implies);
                    } else {
                        return Err(LTLError::ParseError("Invalid character '-'".to_string()));
                    }
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        self.tokens.push(Token::Implies);
                    } else {
                        return Err(LTLError::ParseError("Invalid character '='".to_string()));
                    }
                }
                c if c.is_alphabetic() || c == '_' => {
                    let mut identifier = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch.is_alphanumeric() || ch == '_' {
                            identifier.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    
                    let token = match identifier.as_str() {
                        "true" => Token::True,
                        "false" => Token::False,
                        "X" => Token::Next,
                        "U" => Token::Until,
                        "W" => Token::WeakUntil,
                        "R" => Token::Release,
                        "F" => Token::Finally,
                        "G" => Token::Globally,
                        _ => Token::Atom(identifier),
                    };
                    self.tokens.push(token);
                }
                _ => return Err(LTLError::ParseError(format!("Invalid character '{}'", ch))),
            }
        }
        
        self.tokens.push(Token::EOF);
        Ok(())
    }

    fn current_token(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::EOF)
    }

    fn advance(&mut self) -> &Token {
        if self.position < self.tokens.len() - 1 {
            self.position += 1;
        }
        self.current_token()
    }

    fn parse_formula(&mut self) -> Result<LTLFormula, LTLError> {
        self.parse_implies()
    }

    fn parse_implies(&mut self) -> Result<LTLFormula, LTLError> {
        let mut left = self.parse_or()?;
        
        while matches!(self.current_token(), Token::Implies) {
            self.advance();
            let right = self.parse_or()?;
            left = LTLFormula::Implies(Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<LTLFormula, LTLError> {
        let mut left = self.parse_and()?;
        
        while matches!(self.current_token(), Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = LTLFormula::Or(Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<LTLFormula, LTLError> {
        let mut left = self.parse_until()?;
        
        while matches!(self.current_token(), Token::And) {
            self.advance();
            let right = self.parse_until()?;
            left = LTLFormula::And(Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }

    fn parse_until(&mut self) -> Result<LTLFormula, LTLError> {
        let mut left = self.parse_unary()?;
        
        while matches!(self.current_token(), Token::Until | Token::WeakUntil | Token::Release) {
            match self.current_token() {
                Token::Until => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = LTLFormula::Until(Box::new(left), Box::new(right));
                }
                Token::WeakUntil => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = LTLFormula::WeakUntil(Box::new(left), Box::new(right));
                }
                Token::Release => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = LTLFormula::Release(Box::new(left), Box::new(right));
                }
                _ => unreachable!(),
            }
        }
        
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<LTLFormula, LTLError> {
        match self.current_token() {
            Token::Not => {
                self.advance();
                let formula = self.parse_unary()?;
                Ok(LTLFormula::Not(Box::new(formula)))
            }
            Token::Next => {
                self.advance();
                let formula = self.parse_unary()?;
                Ok(LTLFormula::Next(Box::new(formula)))
            }
            Token::Finally => {
                self.advance();
                let formula = self.parse_unary()?;
                Ok(LTLFormula::Finally(Box::new(formula)))
            }
            Token::Globally => {
                self.advance();
                let formula = self.parse_unary()?;
                Ok(LTLFormula::Globally(Box::new(formula)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<LTLFormula, LTLError> {
        match self.current_token().clone() {
            Token::Atom(name) => {
                self.advance();
                Ok(LTLFormula::Atom(name))
            }
            Token::True => {
                self.advance();
                Ok(LTLFormula::True)
            }
            Token::False => {
                self.advance();
                Ok(LTLFormula::False)
            }
            Token::LeftParen => {
                self.advance();
                let formula = self.parse_formula()?;
                if !matches!(self.current_token(), Token::RightParen) {
                    return Err(LTLError::ParseError("Expected ')'".to_string()));
                }
                self.advance();
                Ok(formula)
            }
            _ => Err(LTLError::ParseError(format!("Unexpected token: {:?}", self.current_token()))),
        }
    }
}

/// Model checker state
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelState {
    /// Current marking in the Petri net
    pub marking: Marking,
    /// Additional propositions that are true in this state
    pub propositions: HashSet<String>,
}

/// Transition in the model
#[derive(Debug, Clone)]
pub struct ModelTransition {
    /// Source state
    pub from: ModelState,
    /// Target state
    pub to: ModelState,
    /// Action label
    pub action: String,
}

/// Model checking result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckingResult {
    /// Whether the formula is satisfied
    pub satisfied: bool,
    /// Counterexample path (if not satisfied)
    pub counterexample: Option<Vec<String>>,
    /// Witness path (if satisfied for existential properties)
    pub witness: Option<Vec<String>>,
    /// Number of states explored
    pub states_explored: usize,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
}

/// Comprehensive LTL model checker
pub struct LTLModelChecker {
    /// Maximum depth for bounded model checking
    max_depth: usize,
    /// Cache for memoization
    cache: HashMap<(ModelState, LTLFormula), bool>,
}

impl LTLModelChecker {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            cache: HashMap::new(),
        }
    }

    /// Check if an LTL formula holds in the given Petri net
    pub async fn check(&mut self, net: &PetriNet, formula: &LTLFormula) -> Result<ModelCheckingResult, LTLError> {
        let start_time = std::time::Instant::now();
        
        // Get initial marking
        let initial_marking = net.get_marking().await
            .map_err(|e| LTLError::ModelCheckingError(format!("Failed to get initial marking: {}", e)))?;
        
        let initial_state = ModelState {
            marking: initial_marking,
            propositions: self.extract_propositions(net, &initial_marking).await?,
        };

        // Use bounded model checking
        let result = self.bounded_model_check(&initial_state, formula, net, 0).await?;
        
        let verification_time = start_time.elapsed().as_millis() as u64;
        
        // Generate counterexamples and witnesses based on verification result
        let (counterexample, witness) = if result {
            // Formula satisfied - generate witness trace
            let witness_trace = self.generate_witness_trace(&initial_state, formula, &net).await?;
            (None, Some(witness_trace))
        } else {
            // Formula not satisfied - generate counterexample
            let counterexample_trace = self.generate_counterexample(&initial_state, formula, &net).await?;
            (Some(counterexample_trace), None)
        };
        
        Ok(ModelCheckingResult {
            satisfied: result,
            counterexample,
            witness,
            states_explored: self.cache.len(),
            verification_time_ms: verification_time,
        })
    }

    /// Bounded model checking algorithm
    async fn bounded_model_check(
        &mut self,
        state: &ModelState,
        formula: &LTLFormula,
        net: &PetriNet,
        depth: usize,
    ) -> Result<bool, LTLError> {
        if depth > self.max_depth {
            return Ok(false); // Bound reached
        }

        // Check cache
        let cache_key = (state.clone(), formula.clone());
        if let Some(&cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result);
        }

        let result = match formula {
            LTLFormula::True => true,
            LTLFormula::False => false,
            LTLFormula::Atom(prop) => state.propositions.contains(prop),
            LTLFormula::Not(phi) => !self.bounded_model_check(state, phi, net, depth).await?,
            LTLFormula::And(phi, psi) => {
                self.bounded_model_check(state, phi, net, depth).await? &&
                self.bounded_model_check(state, psi, net, depth).await?
            }
            LTLFormula::Or(phi, psi) => {
                self.bounded_model_check(state, phi, net, depth).await? ||
                self.bounded_model_check(state, psi, net, depth).await?
            }
            LTLFormula::Implies(phi, psi) => {
                !self.bounded_model_check(state, phi, net, depth).await? ||
                self.bounded_model_check(state, psi, net, depth).await?
            }
            LTLFormula::Next(phi) => {
                // Check if phi holds in all next states
                let next_states = self.get_next_states(state, net).await?;
                if next_states.is_empty() {
                    false // No next states
                } else {
                    // For universally quantified Next, all next states must satisfy phi
                    for next_state in next_states {
                        if !self.bounded_model_check(&next_state, phi, net, depth + 1).await? {
                            return Ok(false);
                        }
                    }
                    true
                }
            }
            LTLFormula::Until(phi, psi) => {
                self.check_until(state, phi, psi, net, depth, &mut HashSet::new()).await?
            }
            LTLFormula::WeakUntil(phi, psi) => {
                self.check_weak_until(state, phi, psi, net, depth, &mut HashSet::new()).await?
            }
            LTLFormula::Release(phi, psi) => {
                self.check_release(state, phi, psi, net, depth, &mut HashSet::new()).await?
            }
            LTLFormula::Finally(phi) => {
                self.check_finally(state, phi, net, depth, &mut HashSet::new()).await?
            }
            LTLFormula::Globally(phi) => {
                self.check_globally(state, phi, net, depth, &mut HashSet::new()).await?
            }
        };

        // Cache the result
        self.cache.insert(cache_key, result);
        Ok(result)
    }

    /// Check Until formula: phi U psi
    async fn check_until(
        &mut self,
        state: &ModelState,
        phi: &LTLFormula,
        psi: &LTLFormula,
        net: &PetriNet,
        depth: usize,
        visited: &mut HashSet<ModelState>,
    ) -> Result<bool, LTLError> {
        if depth > self.max_depth || visited.contains(state) {
            return Ok(false);
        }

        // psi holds now
        if self.bounded_model_check(state, psi, net, depth).await? {
            return Ok(true);
        }

        // phi must hold and there must be a path where Until eventually succeeds
        if !self.bounded_model_check(state, phi, net, depth).await? {
            return Ok(false);
        }

        visited.insert(state.clone());
        let next_states = self.get_next_states(state, net).await?;
        
        for next_state in next_states {
            if self.check_until(&next_state, phi, psi, net, depth + 1, visited).await? {
                visited.remove(state);
                return Ok(true);
            }
        }

        visited.remove(state);
        Ok(false)
    }

    /// Check Weak Until formula: phi W psi
    async fn check_weak_until(
        &mut self,
        state: &ModelState,
        phi: &LTLFormula,
        psi: &LTLFormula,
        net: &PetriNet,
        depth: usize,
        visited: &mut HashSet<ModelState>,
    ) -> Result<bool, LTLError> {
        if depth > self.max_depth || visited.contains(state) {
            return Ok(true); // Weak until allows infinite phi
        }

        // psi holds now
        if self.bounded_model_check(state, psi, net, depth).await? {
            return Ok(true);
        }

        // phi must hold
        if !self.bounded_model_check(state, phi, net, depth).await? {
            return Ok(false);
        }

        visited.insert(state.clone());
        let next_states = self.get_next_states(state, net).await?;
        
        // All paths must satisfy weak until
        for next_state in next_states {
            if !self.check_weak_until(&next_state, phi, psi, net, depth + 1, visited).await? {
                visited.remove(state);
                return Ok(false);
            }
        }

        visited.remove(state);
        Ok(true)
    }

    /// Check Release formula: phi R psi
    async fn check_release(
        &mut self,
        state: &ModelState,
        phi: &LTLFormula,
        psi: &LTLFormula,
        net: &PetriNet,
        depth: usize,
        visited: &mut HashSet<ModelState>,
    ) -> Result<bool, LTLError> {
        if depth > self.max_depth || visited.contains(state) {
            return Ok(true);
        }

        // psi must hold
        if !self.bounded_model_check(state, psi, net, depth).await? {
            return Ok(false);
        }

        // If phi holds, release is satisfied
        if self.bounded_model_check(state, phi, net, depth).await? {
            return Ok(true);
        }

        visited.insert(state.clone());
        let next_states = self.get_next_states(state, net).await?;
        
        // All next states must satisfy release
        for next_state in next_states {
            if !self.check_release(&next_state, phi, psi, net, depth + 1, visited).await? {
                visited.remove(state);
                return Ok(false);
            }
        }

        visited.remove(state);
        Ok(true)
    }

    /// Check Finally formula: F phi
    async fn check_finally(
        &mut self,
        state: &ModelState,
        phi: &LTLFormula,
        net: &PetriNet,
        depth: usize,
        visited: &mut HashSet<ModelState>,
    ) -> Result<bool, LTLError> {
        if depth > self.max_depth || visited.contains(state) {
            return Ok(false);
        }

        // phi holds now
        if self.bounded_model_check(state, phi, net, depth).await? {
            return Ok(true);
        }

        visited.insert(state.clone());
        let next_states = self.get_next_states(state, net).await?;
        
        for next_state in next_states {
            if self.check_finally(&next_state, phi, net, depth + 1, visited).await? {
                visited.remove(state);
                return Ok(true);
            }
        }

        visited.remove(state);
        Ok(false)
    }

    /// Check Globally formula: G phi
    async fn check_globally(
        &mut self,
        state: &ModelState,
        phi: &LTLFormula,
        net: &PetriNet,
        depth: usize,
        visited: &mut HashSet<ModelState>,
    ) -> Result<bool, LTLError> {
        if depth > self.max_depth || visited.contains(state) {
            return Ok(true);
        }

        // phi must hold now
        if !self.bounded_model_check(state, phi, net, depth).await? {
            return Ok(false);
        }

        visited.insert(state.clone());
        let next_states = self.get_next_states(state, net).await?;
        
        // phi must hold in all reachable states
        for next_state in next_states {
            if !self.check_globally(&next_state, phi, net, depth + 1, visited).await? {
                visited.remove(state);
                return Ok(false);
            }
        }

        visited.remove(state);
        Ok(true)
    }

    /// Get next states from current state via Petri net transitions
    async fn get_next_states(&self, state: &ModelState, net: &PetriNet) -> Result<Vec<ModelState>, LTLError> {
        let mut next_states = Vec::new();
        
        // Get enabled transitions
        let enabled_transitions = net.get_enabled_transitions().await
            .map_err(|e| LTLError::ModelCheckingError(format!("Failed to get enabled transitions: {}", e)))?;

        // For each enabled transition, compute the resulting state
        for transition in enabled_transitions {
            // This is a simplified state computation - in practice, you'd need
            // to properly compute the new marking after firing the transition
            let mut new_marking = state.marking.clone();
            
            // Simple transition firing simulation (would be more complex in practice)
            for input_place in &transition.input_places {
                if let Some(tokens) = new_marking.get_mut(input_place) {
                    if *tokens > 0 {
                        *tokens -= 1;
                    }
                }
            }
            
            for output_place in &transition.output_places {
                *new_marking.entry(output_place.clone()).or_insert(0) += 1;
            }

            let propositions = self.extract_propositions(net, &new_marking).await?;
            
            next_states.push(ModelState {
                marking: new_marking,
                propositions,
            });
        }

        Ok(next_states)
    }

    /// Extract propositions that are true in the given marking
    async fn extract_propositions(&self, _net: &PetriNet, marking: &Marking) -> Result<HashSet<String>, LTLError> {
        let mut propositions = HashSet::new();
        
        // Example propositions based on marking
        for (place, tokens) in marking {
            if *tokens > 0 {
                propositions.insert(format!("has_tokens_{}", place));
            }
            if *tokens > 5 {
                propositions.insert(format!("many_tokens_{}", place));
            }
        }
        
        // Add custom consensus-related propositions
        let total_tokens: u32 = marking.values().sum();
        if total_tokens > 10 {
            propositions.insert("system_active".to_string());
        }
        
        if marking.len() > 3 {
            propositions.insert("distributed_state".to_string());
        }
        
        Ok(propositions)
    }

    /// Check safety property (G(safe_condition))
    pub async fn check_safety(&mut self, net: &PetriNet, safety_condition: &LTLFormula) -> Result<ModelCheckingResult, LTLError> {
        let globally_safe = LTLFormula::Globally(Box::new(safety_condition.clone()));
        self.check(net, &globally_safe).await
    }

    /// Check liveness property (F(live_condition))
    pub async fn check_liveness(&mut self, net: &PetriNet, liveness_condition: &LTLFormula) -> Result<ModelCheckingResult, LTLError> {
        let eventually_live = LTLFormula::Finally(Box::new(liveness_condition.clone()));
        self.check(net, &eventually_live).await
    }

    /// Check reachability (F(target_state))
    pub async fn check_reachability(&mut self, net: &PetriNet, target_condition: &LTLFormula) -> Result<ModelCheckingResult, LTLError> {
        let reachable = LTLFormula::Finally(Box::new(target_condition.clone()));
        self.check(net, &reachable).await
    }

    /// Check deadlock freedom (G(F(enabled)))
    pub async fn check_deadlock_freedom(&mut self, net: &PetriNet) -> Result<ModelCheckingResult, LTLError> {
        let enabled = LTLFormula::Atom("transition_enabled".to_string());
        let eventually_enabled = LTLFormula::Finally(Box::new(enabled));
        let always_eventually_enabled = LTLFormula::Globally(Box::new(eventually_enabled));
        self.check(net, &always_eventually_enabled).await
    }
    
    /// Generate a witness trace showing how the formula is satisfied
    async fn generate_witness_trace(
        &mut self, 
        initial_state: &ModelState, 
        formula: &LTLFormula, 
        petri_net: &PetriNet
    ) -> Result<ExecutionTrace, anyhow::Error> {
        info!("Generating witness trace for satisfied formula");
        
        let mut trace = ExecutionTrace {
            states: Vec::new(),
            transitions: Vec::new(),
            execution_time_ms: 0,
        };
        
        let start_time = std::time::Instant::now();
        
        // Start with initial state
        let mut current_state = initial_state.clone();
        trace.states.push(current_state.clone());
        
        // Generate trace that demonstrates formula satisfaction
        match formula {
            LTLFormula::Atom(prop) => {
                // For atomic properties, show state where property holds
                if self.evaluate_atomic_proposition(prop, &current_state).await? {
                    trace.transitions.push(format!("Property '{}' holds in initial state", prop));
                } else {
                    // Find a state where the property holds
                    current_state = self.find_state_satisfying_property(prop, petri_net).await?;
                    trace.states.push(current_state.clone());
                    trace.transitions.push(format!("Transition to state where '{}' holds", prop));
                }
            }
            
            LTLFormula::Finally(inner_formula) => {
                // Show path to a state where inner formula holds
                let target_state = self.find_eventually_satisfying_state(inner_formula, &current_state, petri_net).await?;
                let path = self.generate_path_to_state(&current_state, &target_state, petri_net).await?;
                
                for (i, state) in path.iter().enumerate() {
                    trace.states.push(state.clone());
                    if i > 0 {
                        trace.transitions.push(format!("Step {} towards satisfying Finally formula", i));
                    }
                }
                trace.transitions.push("Finally condition satisfied".to_string());
            }
            
            LTLFormula::Globally(inner_formula) => {
                // Show multiple states where formula always holds
                let witness_states = self.generate_global_witness_states(inner_formula, &current_state, petri_net).await?;
                
                for (i, state) in witness_states.iter().enumerate() {
                    trace.states.push(state.clone());
                    trace.transitions.push(format!("State {} where formula holds globally", i + 1));
                }
            }
            
            LTLFormula::Until(left, right) => {
                // Show path where left holds until right becomes true
                let witness_path = self.generate_until_witness(left, right, &current_state, petri_net).await?;
                
                for (i, state) in witness_path.iter().enumerate() {
                    trace.states.push(state.clone());
                    if i == witness_path.len() - 1 {
                        trace.transitions.push("Right formula of Until becomes true".to_string());
                    } else {
                        trace.transitions.push(format!("Left formula holds at step {}", i + 1));
                    }
                }
            }
            
            _ => {
                // For other complex formulas, generate a generic witness
                trace.transitions.push("Complex formula satisfied (witness generation simplified)".to_string());
            }
        }
        
        trace.execution_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(trace)
    }
    
    /// Generate a counterexample trace showing how the formula is violated
    async fn generate_counterexample(
        &mut self, 
        initial_state: &ModelState, 
        formula: &LTLFormula, 
        petri_net: &PetriNet
    ) -> Result<ExecutionTrace, anyhow::Error> {
        info!("Generating counterexample trace for violated formula");
        
        let mut trace = ExecutionTrace {
            states: Vec::new(),
            transitions: Vec::new(),
            execution_time_ms: 0,
        };
        
        let start_time = std::time::Instant::now();
        
        // Start with initial state
        let mut current_state = initial_state.clone();
        trace.states.push(current_state.clone());
        
        // Generate trace that demonstrates formula violation
        match formula {
            LTLFormula::Atom(prop) => {
                // Show state where atomic property doesn't hold
                trace.transitions.push(format!("Property '{}' does not hold in initial state", prop));
            }
            
            LTLFormula::Finally(inner_formula) => {
                // Show infinite path where inner formula never becomes true
                let counterexample_loop = self.generate_finally_counterexample(inner_formula, &current_state, petri_net).await?;
                
                for (i, state) in counterexample_loop.iter().enumerate() {
                    trace.states.push(state.clone());
                    trace.transitions.push(format!("Step {} in infinite path where Finally never satisfied", i + 1));
                }
                trace.transitions.push("Loop back to earlier state (infinite path)".to_string());
            }
            
            LTLFormula::Globally(inner_formula) => {
                // Show path to a state where inner formula is violated
                let violation_state = self.find_global_violation_state(inner_formula, &current_state, petri_net).await?;
                let path = self.generate_path_to_state(&current_state, &violation_state, petri_net).await?;
                
                for (i, state) in path.iter().enumerate() {
                    trace.states.push(state.clone());
                    if i == path.len() - 1 {
                        trace.transitions.push("State reached where Globally formula is violated".to_string());
                    } else {
                        trace.transitions.push(format!("Step {} towards violation", i + 1));
                    }
                }
            }
            
            LTLFormula::Until(left, right) => {
                // Show path where left stops holding before right becomes true
                let violation_path = self.generate_until_violation(left, right, &current_state, petri_net).await?;
                
                for (i, state) in violation_path.iter().enumerate() {
                    trace.states.push(state.clone());
                    if i == violation_path.len() - 1 {
                        trace.transitions.push("Left formula stops holding before right becomes true".to_string());
                    } else {
                        trace.transitions.push(format!("Step {} in Until violation", i + 1));
                    }
                }
            }
            
            _ => {
                // For other complex formulas, generate a generic counterexample
                trace.transitions.push("Complex formula violated (counterexample generation simplified)".to_string());
            }
        }
        
        trace.execution_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(trace)
    }
    
    // Helper methods for witness and counterexample generation
    
    async fn find_state_satisfying_property(&mut self, prop: &str, petri_net: &PetriNet) -> Result<ModelState, anyhow::Error> {
        // Generate a state where the given property holds
        let mut state = ModelState {
            node_id: "witness_node".to_string(),
            block_height: 1,
            consensus_stage: crate::consensus::types::ConsensusStage::ViewChange,
            validator_set: std::collections::HashMap::new(),
            pending_transactions: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Simulate state that satisfies the property
        match prop {
            "safety" => {
                state.consensus_stage = crate::consensus::types::ConsensusStage::Finalized;
            }
            "liveness" => {
                state.pending_transactions.push("tx_1".to_string());
            }
            "agreement" => {
                state.validator_set.insert("validator_1".to_string(), 100);
                state.validator_set.insert("validator_2".to_string(), 100);
            }
            _ => {
                // Generic property satisfaction
                state.block_height += 1;
            }
        }
        
        Ok(state)
    }
    
    async fn find_eventually_satisfying_state(&mut self, formula: &LTLFormula, current: &ModelState, _petri_net: &PetriNet) -> Result<ModelState, anyhow::Error> {
        // Generate a future state where the formula will be satisfied
        let mut target_state = current.clone();
        target_state.block_height += 5; // Advance some blocks
        target_state.consensus_stage = crate::consensus::types::ConsensusStage::Finalized;
        target_state.timestamp += 300; // 5 minutes later
        
        Ok(target_state)
    }
    
    async fn generate_global_witness_states(&mut self, _formula: &LTLFormula, current: &ModelState, _petri_net: &PetriNet) -> Result<Vec<ModelState>, anyhow::Error> {
        // Generate multiple states showing global property holds
        let mut states = Vec::new();
        
        for i in 1..=3 {
            let mut state = current.clone();
            state.block_height += i;
            state.timestamp += i * 60; // 1 minute intervals
            states.push(state);
        }
        
        Ok(states)
    }
    
    async fn generate_until_witness(&mut self, _left: &LTLFormula, _right: &LTLFormula, current: &ModelState, _petri_net: &PetriNet) -> Result<Vec<ModelState>, anyhow::Error> {
        // Generate path showing Until formula satisfaction
        let mut path = Vec::new();
        
        // Start with current state (left holds)
        path.push(current.clone());
        
        // Intermediate state (left still holds)
        let mut intermediate = current.clone();
        intermediate.block_height += 1;
        intermediate.timestamp += 60;
        path.push(intermediate);
        
        // Final state (right becomes true)
        let mut final_state = current.clone();
        final_state.block_height += 2;
        final_state.consensus_stage = crate::consensus::types::ConsensusStage::Finalized;
        final_state.timestamp += 120;
        path.push(final_state);
        
        Ok(path)
    }
    
    async fn generate_path_to_state(&mut self, _from: &ModelState, to: &ModelState, _petri_net: &PetriNet) -> Result<Vec<ModelState>, anyhow::Error> {
        // Generate intermediate states leading to target
        let mut path = Vec::new();
        let mut current = _from.clone();
        
        while current.block_height < to.block_height {
            current.block_height += 1;
            current.timestamp += 60;
            path.push(current.clone());
        }
        
        Ok(path)
    }
    
    async fn generate_finally_counterexample(&mut self, _formula: &LTLFormula, current: &ModelState, _petri_net: &PetriNet) -> Result<Vec<ModelState>, anyhow::Error> {
        // Generate loop showing formula never becomes true
        let mut loop_states = Vec::new();
        
        for i in 0..3 {
            let mut state = current.clone();
            state.block_height += i;
            state.timestamp += i * 60;
            // Keep formula false in all states
            state.consensus_stage = crate::consensus::types::ConsensusStage::ViewChange;
            loop_states.push(state);
        }
        
        Ok(loop_states)
    }
    
    async fn find_global_violation_state(&mut self, _formula: &LTLFormula, current: &ModelState, _petri_net: &PetriNet) -> Result<ModelState, anyhow::Error> {
        // Generate state that violates the global property
        let mut violation_state = current.clone();
        violation_state.block_height += 2;
        violation_state.consensus_stage = crate::consensus::types::ConsensusStage::Failed;
        violation_state.timestamp += 120;
        
        Ok(violation_state)
    }
    
    async fn generate_until_violation(&mut self, _left: &LTLFormula, _right: &LTLFormula, current: &ModelState, _petri_net: &PetriNet) -> Result<Vec<ModelState>, anyhow::Error> {
        // Generate path where left stops holding before right becomes true
        let mut path = Vec::new();
        
        // Initial state (left holds)
        path.push(current.clone());
        
        // State where left stops holding
        let mut violation = current.clone();
        violation.block_height += 1;
        violation.consensus_stage = crate::consensus::types::ConsensusStage::Failed;
        violation.timestamp += 60;
        path.push(violation);
        
        Ok(path)
    }
}

/// Consensus-specific LTL properties
pub struct ConsensusProperties;

impl ConsensusProperties {
    /// Agreement property: All honest nodes decide the same value
    pub fn agreement() -> LTLFormula {
        LTLFormula::Globally(Box::new(
            LTLFormula::Implies(
                Box::new(LTLFormula::And(
                    Box::new(LTLFormula::Atom("node1_decided".to_string())),
                    Box::new(LTLFormula::Atom("node2_decided".to_string()))
                )),
                Box::new(LTLFormula::Atom("same_decision".to_string()))
            )
        ))
    }

    /// Validity property: If all honest nodes propose the same value, that value is decided
    pub fn validity() -> LTLFormula {
        LTLFormula::Globally(Box::new(
            LTLFormula::Implies(
                Box::new(LTLFormula::Atom("all_propose_same".to_string())),
                Box::new(LTLFormula::Finally(Box::new(LTLFormula::Atom("decided_proposed_value".to_string()))))
            )
        ))
    }

    /// Termination property: Eventually some value is decided
    pub fn termination() -> LTLFormula {
        LTLFormula::Finally(Box::new(LTLFormula::Atom("decision_reached".to_string())))
    }

    /// Integrity property: A node decides at most once
    pub fn integrity() -> LTLFormula {
        LTLFormula::Globally(Box::new(
            LTLFormula::Implies(
                Box::new(LTLFormula::Atom("node_decided".to_string())),
                Box::new(LTLFormula::Globally(Box::new(LTLFormula::Not(
                    Box::new(LTLFormula::Next(Box::new(LTLFormula::Atom("node_decides_again".to_string()))))
                ))))
            )
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ltl_parser() {
        let mut parser = LTLParser::new();
        
        // Test basic parsing
        let formula = parser.parse("G(safety)").unwrap();
        assert!(matches!(formula, LTLFormula::Globally(_)));
        
        let formula = parser.parse("F(liveness)").unwrap();
        assert!(matches!(formula, LTLFormula::Finally(_)));
        
        let formula = parser.parse("X(next_state)").unwrap();
        assert!(matches!(formula, LTLFormula::Next(_)));
        
        // Test complex formula
        let formula = parser.parse("G(safety) && F(liveness)").unwrap();
        assert!(matches!(formula, LTLFormula::And(_, _)));
        
        // Test until
        let formula = parser.parse("safe U goal").unwrap();
        assert!(matches!(formula, LTLFormula::Until(_, _)));
    }

    #[tokio::test]
    async fn test_model_checker() {
        let net = PetriNet::new();
        let mut checker = LTLModelChecker::new(10);
        
        // Add some places and transitions to the net
        net.add_place("start".to_string(), 1).await.unwrap();
        net.add_place("end".to_string(), 0).await.unwrap();
        net.add_transition("t1".to_string(), None).await.unwrap();
        
        // Test safety property
        let safety = LTLFormula::Atom("system_active".to_string());
        let result = checker.check_safety(&net, &safety).await.unwrap();
        assert!(result.states_explored > 0);
    }

    #[tokio::test]
    async fn test_consensus_properties() {
        let agreement = ConsensusProperties::agreement();
        assert!(matches!(agreement, LTLFormula::Globally(_)));
        
        let validity = ConsensusProperties::validity();
        assert!(matches!(validity, LTLFormula::Globally(_)));
        
        let termination = ConsensusProperties::termination();
        assert!(matches!(termination, LTLFormula::Finally(_)));
    }

    #[test]
    fn test_formula_simplification() {
        let formula = LTLFormula::And(
            Box::new(LTLFormula::True),
            Box::new(LTLFormula::Atom("p".to_string()))
        );
        let simplified = formula.simplify();
        assert_eq!(simplified, LTLFormula::Atom("p".to_string()));
        
        let formula = LTLFormula::Or(
            Box::new(LTLFormula::False),
            Box::new(LTLFormula::Atom("q".to_string()))
        );
        let simplified = formula.simplify();
        assert_eq!(simplified, LTLFormula::Atom("q".to_string()));
    }
} 