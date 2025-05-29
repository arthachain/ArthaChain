use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::fmt::Debug;
use serde::{Serialize, Deserialize};
// use z3::{Context, Config, Solver, Ast, Sort, FuncDecl, Model};  // Temporarily disabled
use tokio::sync::RwLock as TokioRwLock;
use anyhow::Result;

/// Place in Petri net
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub struct Place {
    /// Place name
    pub name: String,
    /// Place type
    pub place_type: PlaceType,
    /// Initial tokens
    pub initial_tokens: u32,
    /// Maximum tokens
    pub max_tokens: Option<u32>,
}

/// Place type
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum PlaceType {
    /// Regular place
    Regular,
    /// Input place
    Input,
    /// Output place
    Output,
    /// Control place
    Control,
}

/// Transition in Petri net
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transition {
    /// Transition name
    pub id: String,
    /// Input places
    pub input_places: HashSet<String>,
    /// Output places
    pub output_places: HashSet<String>,
    /// Guard condition
    pub guard: Option<GuardCondition>,
    /// Priority (higher values have higher priority)
    pub priority: u32,
}

/// Guard condition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct GuardCondition {
    pub condition_type: GuardType,
    pub threshold: u32,
    pub current_value: u32,
}

impl GuardCondition {
    pub fn new(condition_type: GuardType, threshold: u32) -> Self {
        Self {
            condition_type,
            threshold,
            current_value: 0,
        }
    }

    pub fn evaluate(&self) -> bool {
        match self.condition_type {
            GuardType::GreaterThanOrEqual => self.current_value >= self.threshold,
            GuardType::LessThan => self.current_value < self.threshold,
            GuardType::Equal => self.current_value == self.threshold,
            GuardType::NotEqual => self.current_value != self.threshold,
        }
    }

    pub fn update(&mut self, value: u32) {
        self.current_value = value;
    }
}

/// Guard type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum GuardType {
    GreaterThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
}

/// Arc in Petri net
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetriArc {
    /// Source place
    pub source: String,
    /// Target transition
    pub target: String,
    /// Weight
    pub weight: u32,
    /// Arc type
    pub arc_type: ArcType,
}

/// Arc type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArcType {
    /// Normal arc
    Normal,
    /// Inhibitor arc (transition enabled only if place has no tokens)
    Inhibitor,
    /// Reset arc (removes all tokens from place)
    Reset,
}

/// Marking in Petri net
pub type Marking = HashMap<String, u32>;

/// Firing sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiringSequence {
    /// Sequence of transition IDs
    pub transitions: Vec<String>,
    /// Markings after each transition
    pub markings: Vec<Marking>,
    /// Total cost/time
    pub cost: u32,
}

/// Reachability analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityResult {
    /// Whether the target is reachable
    pub is_reachable: bool,
    /// Shortest path to target
    pub shortest_path: Option<FiringSequence>,
    /// All reachable markings
    pub reachable_markings: Vec<Marking>,
    /// Number of states explored
    pub states_explored: usize,
    /// Analysis time in milliseconds
    pub analysis_time_ms: u64,
}

/// Petri net
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetriNet {
    #[serde(with = "tokio_serde")]
    places: Arc<TokioRwLock<HashMap<String, Place>>>,
    #[serde(with = "tokio_serde")]
    transitions: Arc<TokioRwLock<HashMap<String, Transition>>>,
    #[serde(with = "tokio_serde")]
    arcs: Arc<TokioRwLock<Vec<PetriArc>>>,
    /// Maximum bound for reachability analysis
    pub max_bound: u32,
}

/// Analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalysisResult {
    /// Whether the net is bounded
    pub is_bounded: bool,
    /// Whether the net is deadlock-free
    pub is_deadlock_free: bool,
    /// Whether the initial marking is reachable
    pub is_reachable: bool,
    /// Reachable markings
    pub reachable_markings: Vec<Marking>,
    /// Whether the net is safe (1-bounded)
    pub is_safe: bool,
    /// Whether the net is live
    pub is_live: bool,
    /// Strongly connected components
    pub scc_count: usize,
    /// Analysis statistics
    pub stats: AnalysisStats,
}

/// Analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalysisStats {
    /// Number of places
    pub places_count: usize,
    /// Number of transitions
    pub transitions_count: usize,
    /// Number of arcs
    pub arcs_count: usize,
    /// Total reachable states
    pub reachable_states: usize,
    /// Maximum marking bound
    pub max_marking_bound: u32,
    /// Analysis duration in milliseconds
    pub analysis_duration_ms: u64,
}

/// Custom serialization module for tokio types
mod tokio_serde {
    use super::*;
    use serde::{Serializer, Deserializer};

    pub fn serialize<S, T>(value: &Arc<TokioRwLock<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Send + Sync,
    {
        let guard = value.try_read().map_err(serde::ser::Error::custom)?;
        T::serialize(&*guard, serializer)
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Arc<TokioRwLock<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de> + Send + Sync + 'static,
    {
        let value = T::deserialize(deserializer)?;
        Ok(Arc::new(TokioRwLock::new(value)))
    }
}

impl PetriNet {
    /// Create a new Petri net
    pub fn new() -> Self {
        Self {
            places: Arc::new(TokioRwLock::new(HashMap::new())),
            transitions: Arc::new(TokioRwLock::new(HashMap::new())),
            arcs: Arc::new(TokioRwLock::new(Vec::new())),
            max_bound: 1000,
        }
    }

    /// Create a new bounded Petri net
    pub fn new_bounded(max_bound: u32) -> Self {
        Self {
            places: Arc::new(TokioRwLock::new(HashMap::new())),
            transitions: Arc::new(TokioRwLock::new(HashMap::new())),
            arcs: Arc::new(TokioRwLock::new(Vec::new())),
            max_bound,
        }
    }

    /// Add a place
    pub async fn add_place(&self, name: String, tokens: u32) -> Result<()> {
        let mut places = self.places.write().await;
        places.insert(name.clone(), Place {
            name,
            place_type: PlaceType::Regular,
            initial_tokens: tokens,
            max_tokens: None,
        });
        Ok(())
    }

    /// Add a place with specific type and bounds
    pub async fn add_place_with_type(&self, name: String, tokens: u32, place_type: PlaceType, max_tokens: Option<u32>) -> Result<()> {
        let mut places = self.places.write().await;
        places.insert(name.clone(), Place {
            name,
            place_type,
            initial_tokens: tokens,
            max_tokens,
        });
        Ok(())
    }

    /// Add a transition
    pub async fn add_transition(&self, name: String, guard: Option<GuardCondition>) -> Result<()> {
        let mut transitions = self.transitions.write().await;
        transitions.insert(name.clone(), Transition {
            id: name,
            input_places: HashSet::new(),
            output_places: HashSet::new(),
            guard,
            priority: 0,
        });
        Ok(())
    }

    /// Add a transition with priority
    pub async fn add_transition_with_priority(&self, name: String, guard: Option<GuardCondition>, priority: u32) -> Result<()> {
        let mut transitions = self.transitions.write().await;
        transitions.insert(name.clone(), Transition {
            id: name,
            input_places: HashSet::new(),
            output_places: HashSet::new(),
            guard,
            priority,
        });
        Ok(())
    }

    /// Add an arc
    pub async fn add_arc(&self, from: String, to: String, weight: u32) -> Result<()> {
        let mut arcs = self.arcs.write().await;
        arcs.push(PetriArc {
            source: from,
            target: to,
            weight,
            arc_type: ArcType::Normal,
        });
        Ok(())
    }

    /// Add an arc with specific type
    pub async fn add_arc_with_type(&self, from: String, to: String, weight: u32, arc_type: ArcType) -> Result<()> {
        let mut arcs = self.arcs.write().await;
        arcs.push(PetriArc {
            source: from,
            target: to,
            weight,
            arc_type,
        });
        Ok(())
    }

    /// Get marking
    pub async fn get_marking(&self) -> Result<Marking> {
        let places = self.places.read().await;
        Ok(places.iter().map(|(id, place)| (id.clone(), place.initial_tokens)).collect())
    }

    /// Set marking
    pub async fn set_marking(&self, marking: &Marking) -> Result<()> {
        let mut places = self.places.write().await;
        for (id, tokens) in marking {
            if let Some(place) = places.get_mut(id) {
                place.initial_tokens = *tokens;
            } else {
                anyhow::bail!("Place {} not found", id);
            }
        }
        Ok(())
    }

    /// Check if a transition is enabled
    pub async fn is_enabled(&self, transition: &Transition) -> Result<bool> {
        self.is_enabled_in_marking(transition, &self.get_marking().await?).await
    }

    /// Check if a transition is enabled in a specific marking
    pub async fn is_enabled_in_marking(&self, transition: &Transition, marking: &Marking) -> Result<bool> {
        // Check guard condition
        if let Some(guard) = &transition.guard {
            if !self.evaluate_guard(guard) {
                return Ok(false);
            }
        }

        let arcs = self.arcs.read().await;
        
        // Check input places have sufficient tokens
        for input_place in &transition.input_places {
            let current_tokens = marking.get(input_place).copied().unwrap_or(0);
            
            // Find the arc from this place to the transition
            let arc = arcs.iter()
                .find(|arc| arc.source == *input_place && arc.target == transition.id);
            
            if let Some(arc) = arc {
                match arc.arc_type {
                    ArcType::Normal => {
                        if current_tokens < arc.weight {
                            return Ok(false);
                        }
                    }
                    ArcType::Inhibitor => {
                        if current_tokens >= arc.weight {
                            return Ok(false);
                        }
                    }
                    ArcType::Reset => {
                        // Reset arcs don't affect enablement
                    }
                }
            } else {
                // Default weight of 1 if no arc found
                if current_tokens < 1 {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Evaluate guard condition
    fn evaluate_guard(&self, guard: &GuardCondition) -> bool {
        guard.evaluate()
    }

    /// Fire a transition
    pub async fn fire_transition(&self, transition: &Transition) -> Result<()> {
        if !self.is_enabled(transition).await? {
            anyhow::bail!("Transition {} is not enabled", transition.id);
        }

        let mut marking = self.get_marking().await?;
        self.fire_transition_in_marking(transition, &mut marking).await?;
        self.set_marking(&marking).await?;
        Ok(())
    }

    /// Fire a transition in a specific marking
    pub async fn fire_transition_in_marking(&self, transition: &Transition, marking: &mut Marking) -> Result<()> {
        if !self.is_enabled_in_marking(transition, marking).await? {
            anyhow::bail!("Transition {} is not enabled in the given marking", transition.id);
        }

        let arcs = self.arcs.read().await;

        // Remove tokens from input places
        for input_place in &transition.input_places {
            let arc = arcs.iter()
                .find(|arc| arc.source == *input_place && arc.target == transition.id);
            
            if let Some(arc) = arc {
                match arc.arc_type {
                    ArcType::Normal => {
                        let current = marking.get_mut(input_place).unwrap_or(&mut 0);
                        *current = current.saturating_sub(arc.weight);
                    }
                    ArcType::Reset => {
                        marking.insert(input_place.clone(), 0);
                    }
                    ArcType::Inhibitor => {
                        // Inhibitor arcs don't consume tokens
                    }
                }
            } else {
                // Default: consume 1 token
                let current = marking.get_mut(input_place).unwrap_or(&mut 0);
                *current = current.saturating_sub(1);
            }
        }

        // Add tokens to output places
        for output_place in &transition.output_places {
            let arc = arcs.iter()
                .find(|arc| arc.source == transition.id && arc.target == *output_place);
            
            if let Some(arc) = arc {
                let current = marking.entry(output_place.clone()).or_insert(0);
                *current = current.saturating_add(arc.weight);
            } else {
                // Default: add 1 token
                let current = marking.entry(output_place.clone()).or_insert(0);
                *current = current.saturating_add(1);
            }
        }

        Ok(())
    }

    /// Get enabled transitions
    pub async fn get_enabled_transitions(&self) -> Result<Vec<Transition>> {
        self.get_enabled_transitions_in_marking(&self.get_marking().await?).await
    }

    /// Get enabled transitions in a specific marking
    pub async fn get_enabled_transitions_in_marking(&self, marking: &Marking) -> Result<Vec<Transition>> {
        let transitions = self.transitions.read().await;
        let mut enabled = Vec::new();
        
        for transition in transitions.values() {
            if self.is_enabled_in_marking(transition, marking).await? {
                enabled.push(transition.clone());
            }
        }
        
        // Sort by priority (higher priority first)
        enabled.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        Ok(enabled)
    }

    /// Check if a marking is reachable from the initial marking
    pub async fn is_reachable(&self, target_marking: &Marking) -> Result<bool> {
        let result = self.reachability_analysis(Some(target_marking.clone())).await?;
        Ok(result.is_reachable)
    }

    /// Comprehensive reachability analysis
    pub async fn reachability_analysis(&self, target_marking: Option<Marking>) -> Result<ReachabilityResult> {
        let start_time = std::time::Instant::now();
        let initial_marking = self.get_marking().await?;
        
        let mut reachable_markings = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map = HashMap::new();
        
        queue.push_back(initial_marking.clone());
        visited.insert(initial_marking.clone());
        reachable_markings.push(initial_marking.clone());
        
        let mut target_found = false;
        let mut target_path = None;
        
        while let Some(current_marking) = queue.pop_front() {
            // Check if we found the target
            if let Some(ref target) = target_marking {
                if current_marking == *target {
                    target_found = true;
                    target_path = Some(self.reconstruct_path(&parent_map, &initial_marking, &current_marking).await?);
                    break;
                }
            }
            
            // Get enabled transitions
            let enabled_transitions = self.get_enabled_transitions_in_marking(&current_marking).await?;
            
            for transition in enabled_transitions {
                let mut new_marking = current_marking.clone();
                self.fire_transition_in_marking(&transition, &mut new_marking).await?;
                
                // Check bounds
                if self.is_marking_bounded(&new_marking) && !visited.contains(&new_marking) {
                    visited.insert(new_marking.clone());
                    reachable_markings.push(new_marking.clone());
                    queue.push_back(new_marking.clone());
                    parent_map.insert(new_marking.clone(), (current_marking.clone(), transition.id));
                }
            }
            
            // Prevent infinite loops in unbounded nets
            if visited.len() > 10000 {
                break;
            }
        }
        
        let analysis_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ReachabilityResult {
            is_reachable: target_found,
            shortest_path: target_path,
            reachable_markings,
            states_explored: visited.len(),
            analysis_time_ms: analysis_time,
        })
    }

    /// Check if marking is within bounds
    fn is_marking_bounded(&self, marking: &Marking) -> bool {
        marking.values().all(|&tokens| tokens <= self.max_bound)
    }

    /// Reconstruct path from parent map
    async fn reconstruct_path(&self, parent_map: &HashMap<Marking, (Marking, String)>, initial: &Marking, target: &Marking) -> Result<FiringSequence> {
        let mut path = Vec::new();
        let mut markings = Vec::new();
        let mut current = target.clone();
        
        while current != *initial {
            if let Some((parent, transition_id)) = parent_map.get(&current) {
                path.push(transition_id.clone());
                markings.push(current.clone());
                current = parent.clone();
            } else {
                break;
            }
        }
        
        path.reverse();
        markings.reverse();
        markings.insert(0, initial.clone());
        
        Ok(FiringSequence {
            transitions: path,
            markings,
            cost: markings.len() as u32,
        })
    }

    /// Check if the net is bounded
    pub async fn is_bounded(&self, bound: u32) -> Result<bool> {
        let reachability_result = self.reachability_analysis(None).await?;
        Ok(reachability_result.reachable_markings.iter()
            .all(|marking| marking.values().all(|&tokens| tokens <= bound)))
    }

    /// Check if the net is safe (1-bounded)
    pub async fn is_safe(&self) -> Result<bool> {
        self.is_bounded(1).await
    }

    /// Check if the net is deadlock-free
    pub async fn is_deadlock_free(&self) -> Result<bool> {
        let reachability_result = self.reachability_analysis(None).await?;
        
        // Check if any reachable marking has no enabled transitions
        for marking in &reachability_result.reachable_markings {
            let enabled = self.get_enabled_transitions_in_marking(marking).await?;
            if enabled.is_empty() {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    /// Check if the net is live
    pub async fn is_live(&self) -> Result<bool> {
        let transitions = self.transitions.read().await;
        let reachability_result = self.reachability_analysis(None).await?;
        
        // For each transition, check if it's enabled in at least one reachable marking
        for transition in transitions.values() {
            let mut enabled_somewhere = false;
            for marking in &reachability_result.reachable_markings {
                if self.is_enabled_in_marking(transition, marking).await? {
                    enabled_somewhere = true;
                    break;
                }
            }
            if !enabled_somewhere {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    /// Get reachable markings
    pub async fn get_reachable_markings(&self) -> Result<Vec<Marking>> {
        let result = self.reachability_analysis(None).await?;
        Ok(result.reachable_markings)
    }

    /// Get firing sequence to reach marking
    pub async fn get_firing_sequence(&self, target_marking: &Marking) -> Result<Option<FiringSequence>> {
        let result = self.reachability_analysis(Some(target_marking.clone())).await?;
        Ok(result.shortest_path)
    }

    /// Add connection between places and transitions
    pub async fn connect_places_to_transition(&self, transition_id: &str, input_places: Vec<String>, output_places: Vec<String>) -> Result<()> {
        let mut transitions = self.transitions.write().await;
        if let Some(transition) = transitions.get_mut(transition_id) {
            for place in input_places {
                transition.input_places.insert(place);
            }
            for place in output_places {
                transition.output_places.insert(place);
            }
        } else {
            anyhow::bail!("Transition {} not found", transition_id);
        }
        Ok(())
    }

    /// Get statistics about the net
    pub async fn get_stats(&self) -> Result<AnalysisStats> {
        let start_time = std::time::Instant::now();
        
        let places = self.places.read().await;
        let transitions = self.transitions.read().await;
        let arcs = self.arcs.read().await;
        
        let reachability_result = self.reachability_analysis(None).await?;
        let max_bound = reachability_result.reachable_markings.iter()
            .flat_map(|marking| marking.values())
            .max()
            .copied()
            .unwrap_or(0);
        
        let analysis_duration = start_time.elapsed().as_millis() as u64;
        
        Ok(AnalysisStats {
            places_count: places.len(),
            transitions_count: transitions.len(),
            arcs_count: arcs.len(),
            reachable_states: reachability_result.reachable_markings.len(),
            max_marking_bound: max_bound,
            analysis_duration_ms: analysis_duration,
        })
    }

    /// Create consensus workflow Petri net
    pub async fn create_consensus_workflow() -> Result<Self> {
        let net = Self::new_bounded(100);
        
        // Add places for consensus phases
        net.add_place("init".to_string(), 1).await?;
        net.add_place("propose".to_string(), 0).await?;
        net.add_place("vote".to_string(), 0).await?;
        net.add_place("commit".to_string(), 0).await?;
        net.add_place("finalize".to_string(), 0).await?;
        
        // Add control places
        net.add_place_with_type("validator_ready".to_string(), 3, PlaceType::Control, Some(10)).await?;
        net.add_place_with_type("quorum_reached".to_string(), 0, PlaceType::Control, Some(1)).await?;
        
        // Add transitions for consensus workflow
        net.add_transition_with_priority("start_proposal".to_string(), None, 10).await?;
        net.add_transition_with_priority("collect_votes".to_string(), 
            Some(GuardCondition::new(GuardType::GreaterThanOrEqual, 2)), 5).await?;
        net.add_transition_with_priority("reach_consensus".to_string(),
            Some(GuardCondition::new(GuardType::GreaterThanOrEqual, 2)), 5).await?;
        net.add_transition("finalize_block".to_string(), None).await?;
        
        // Connect workflow
        net.connect_places_to_transition("start_proposal", 
            vec!["init".to_string(), "validator_ready".to_string()], 
            vec!["propose".to_string()]).await?;
        
        net.connect_places_to_transition("collect_votes",
            vec!["propose".to_string(), "validator_ready".to_string()],
            vec!["vote".to_string(), "quorum_reached".to_string()]).await?;
        
        net.connect_places_to_transition("reach_consensus",
            vec!["vote".to_string(), "quorum_reached".to_string()],
            vec!["commit".to_string()]).await?;
        
        net.connect_places_to_transition("finalize_block",
            vec!["commit".to_string()],
            vec!["finalize".to_string()]).await?;
        
        // Add arcs
        net.add_arc("init".to_string(), "start_proposal".to_string(), 1).await?;
        net.add_arc("validator_ready".to_string(), "start_proposal".to_string(), 1).await?;
        net.add_arc("start_proposal".to_string(), "propose".to_string(), 1).await?;
        
        net.add_arc("propose".to_string(), "collect_votes".to_string(), 1).await?;
        net.add_arc("validator_ready".to_string(), "collect_votes".to_string(), 2).await?;
        net.add_arc("collect_votes".to_string(), "vote".to_string(), 1).await?;
        net.add_arc("collect_votes".to_string(), "quorum_reached".to_string(), 1).await?;
        
        net.add_arc("vote".to_string(), "reach_consensus".to_string(), 1).await?;
        net.add_arc("quorum_reached".to_string(), "reach_consensus".to_string(), 1).await?;
        net.add_arc("reach_consensus".to_string(), "commit".to_string(), 1).await?;
        
        net.add_arc("commit".to_string(), "finalize_block".to_string(), 1).await?;
        net.add_arc("finalize_block".to_string(), "finalize".to_string(), 1).await?;
        
        Ok(net)
    }
}

/// Petri net analyzer with advanced capabilities
pub struct PetriNetAnalyzer {
    /// Maximum analysis depth
    max_depth: usize,
}

impl PetriNetAnalyzer {
    /// Create a new Petri net analyzer
    pub fn new() -> Self {
        Self {
            max_depth: 1000,
        }
    }

    /// Create analyzer with custom depth limit
    pub fn new_with_depth(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Comprehensive analysis of Petri net
    pub async fn analyze(&self, net: &PetriNet) -> Result<AnalysisResult> {
        let start_time = std::time::Instant::now();
        let mut result = AnalysisResult::default();

        // Basic structural analysis
        result.stats = net.get_stats().await?;
        
        // Behavioral analysis
        result.is_bounded = net.is_bounded(net.max_bound).await?;
        result.is_safe = net.is_safe().await?;
        result.is_deadlock_free = net.is_deadlock_free().await?;
        result.is_live = net.is_live().await?;
        
        // Reachability analysis
        let reachability_result = net.reachability_analysis(None).await?;
        result.is_reachable = true; // Initial marking is always reachable
        result.reachable_markings = reachability_result.reachable_markings;
        
        // Update analysis duration
        result.stats.analysis_duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(result)
    }

    /// Analyze specific consensus properties
    pub async fn analyze_consensus_properties(&self, net: &PetriNet) -> Result<ConsensusAnalysisResult> {
        let analysis = self.analyze(net).await?;
        
        // Check consensus-specific properties
        let has_finalization = analysis.reachable_markings.iter()
            .any(|marking| marking.get("finalize").unwrap_or(&0) > &0);
        
        let has_deadlock_in_voting = analysis.reachable_markings.iter()
            .any(|marking| {
                marking.get("vote").unwrap_or(&0) > &0 && 
                marking.get("quorum_reached").unwrap_or(&0) == &0
            });
        
        let max_concurrent_validators = analysis.reachable_markings.iter()
            .map(|marking| marking.get("validator_ready").unwrap_or(&0))
            .max()
            .copied()
            .unwrap_or(0);
        
        Ok(ConsensusAnalysisResult {
            basic_analysis: analysis,
            can_reach_finalization: has_finalization,
            has_voting_deadlock: has_deadlock_in_voting,
            max_concurrent_validators,
            consensus_safety: !has_deadlock_in_voting,
            consensus_liveness: has_finalization,
        })
    }
}

/// Consensus-specific analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusAnalysisResult {
    /// Basic Petri net analysis
    pub basic_analysis: AnalysisResult,
    /// Can reach finalization state
    pub can_reach_finalization: bool,
    /// Has deadlock in voting phase
    pub has_voting_deadlock: bool,
    /// Maximum concurrent validators
    pub max_concurrent_validators: u32,
    /// Consensus safety property
    pub consensus_safety: bool,
    /// Consensus liveness property
    pub consensus_liveness: bool,
}

impl Default for PetriNet {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PetriNetAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_petri_net_basic() {
        let net = PetriNet::new();
        
        // Add places
        net.add_place("p1".to_string(), 1).await.unwrap();
        net.add_place("p2".to_string(), 0).await.unwrap();
        
        // Add transition
        net.add_transition("t1".to_string(), None).await.unwrap();
        
        // Connect places to transition
        net.connect_places_to_transition("t1", vec!["p1".to_string()], vec!["p2".to_string()]).await.unwrap();
        
        // Add arcs
        net.add_arc("p1".to_string(), "t1".to_string(), 1).await.unwrap();
        net.add_arc("t1".to_string(), "p2".to_string(), 1).await.unwrap();
        
        // Test enablement
        let transitions = net.transitions.read().await;
        let t1 = transitions.get("t1").unwrap();
        assert!(net.is_enabled(t1).await.unwrap());
        
        // Test firing
        net.fire_transition(t1).await.unwrap();
        
        // Check new marking
        let marking = net.get_marking().await.unwrap();
        assert_eq!(marking.get("p1").unwrap_or(&0), &0);
        assert_eq!(marking.get("p2").unwrap_or(&0), &1);
    }
    
    #[tokio::test]
    async fn test_reachability_analysis() {
        let net = PetriNet::new_bounded(10);
        
        // Create simple workflow
        net.add_place("start".to_string(), 1).await.unwrap();
        net.add_place("middle".to_string(), 0).await.unwrap();
        net.add_place("end".to_string(), 0).await.unwrap();
        
        net.add_transition("t1".to_string(), None).await.unwrap();
        net.add_transition("t2".to_string(), None).await.unwrap();
        
        net.connect_places_to_transition("t1", vec!["start".to_string()], vec!["middle".to_string()]).await.unwrap();
        net.connect_places_to_transition("t2", vec!["middle".to_string()], vec!["end".to_string()]).await.unwrap();
        
        // Add arcs
        net.add_arc("start".to_string(), "t1".to_string(), 1).await.unwrap();
        net.add_arc("t1".to_string(), "middle".to_string(), 1).await.unwrap();
        net.add_arc("middle".to_string(), "t2".to_string(), 1).await.unwrap();
        net.add_arc("t2".to_string(), "end".to_string(), 1).await.unwrap();
        
        // Test reachability
        let mut target_marking = HashMap::new();
        target_marking.insert("end".to_string(), 1);
        target_marking.insert("start".to_string(), 0);
        target_marking.insert("middle".to_string(), 0);
        
        let result = net.reachability_analysis(Some(target_marking)).await.unwrap();
        assert!(result.is_reachable);
        assert!(result.shortest_path.is_some());
        
        let path = result.shortest_path.unwrap();
        assert_eq!(path.transitions.len(), 2);
        assert_eq!(path.transitions[0], "t1");
        assert_eq!(path.transitions[1], "t2");
    }
    
    #[tokio::test]
    async fn test_consensus_workflow() {
        let net = PetriNet::create_consensus_workflow().await.unwrap();
        
        // Test initial state
        let marking = net.get_marking().await.unwrap();
        assert_eq!(marking.get("init").unwrap_or(&0), &1);
        assert_eq!(marking.get("validator_ready").unwrap_or(&0), &3);
        
        // Test reachability to finalization
        let mut target_marking = HashMap::new();
        target_marking.insert("finalize".to_string(), 1);
        
        let is_reachable = net.is_reachable(&target_marking).await.unwrap();
        assert!(is_reachable);
        
        // Analyze consensus properties
        let analyzer = PetriNetAnalyzer::new();
        let consensus_result = analyzer.analyze_consensus_properties(&net).await.unwrap();
        assert!(consensus_result.can_reach_finalization);
        assert!(consensus_result.consensus_safety);
        assert!(consensus_result.consensus_liveness);
    }
    
    #[tokio::test]
    async fn test_guard_conditions() {
        let mut guard = GuardCondition::new(GuardType::GreaterThanOrEqual, 3);
        assert!(!guard.evaluate());
        
        guard.update(3);
        assert!(guard.evaluate());
        
        guard.update(5);
        assert!(guard.evaluate());
        
        guard.update(2);
        assert!(!guard.evaluate());
    }
    
    #[tokio::test]
    async fn test_inhibitor_arcs() {
        let net = PetriNet::new();
        
        net.add_place("p1".to_string(), 1).await.unwrap();
        net.add_place("p2".to_string(), 1).await.unwrap();
        net.add_place("p3".to_string(), 0).await.unwrap();
        
        net.add_transition("t1".to_string(), None).await.unwrap();
        net.connect_places_to_transition("t1", vec!["p1".to_string(), "p2".to_string()], vec!["p3".to_string()]).await.unwrap();
        
        // Add normal arc from p1 and inhibitor arc from p2
        net.add_arc_with_type("p1".to_string(), "t1".to_string(), 1, ArcType::Normal).await.unwrap();
        net.add_arc_with_type("p2".to_string(), "t1".to_string(), 1, ArcType::Inhibitor).await.unwrap();
        net.add_arc("t1".to_string(), "p3".to_string(), 1).await.unwrap();
        
        // Transition should not be enabled because p2 has tokens (inhibitor)
        let transitions = net.transitions.read().await;
        let t1 = transitions.get("t1").unwrap();
        assert!(!net.is_enabled(t1).await.unwrap());
        
        // Remove token from p2
        let mut marking = net.get_marking().await.unwrap();
        marking.insert("p2".to_string(), 0);
        net.set_marking(&marking).await.unwrap();
        
        // Now transition should be enabled
        assert!(net.is_enabled(t1).await.unwrap());
    }
} 