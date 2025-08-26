# Overviews

!!! goal
    Hi! This page is intended for user to understand the overall concept of `inspeqtor` in the high-level first. You might find this page useful when you reviews the interaction between modules and functions offered by `inspeqtor`.

We catergorized characterization and calibration of the quantum device into multiple phase. You might not necessary needs to do or understand every phases and chose to work with specific phases.

- [Experimental Phase](#experimental-phase) is a preparation of the characterization of the quantum device. It might dictate the constraint of your control calibration too.
- [Characterization Phase](#characterization-phase)
- [Control Calibration Phase](#control-calibration-phase)

!!! note
    We would like to remind user that `inspeqtor` is a framework. We provide some opinions of how to do things in the characterization and calibration task. Thus, `inspeqtor` provides user with a varities of utility functions which is designed to be easily replaced by custom function from the user. You don't have to use everything, just what you need 😉

## Experimental Phase

Diagram below shows the sequence of interaction between, the user, `inspeqtor`, quantum device, and file system. Please refer to tutorial of how to work with data and experiment using `inspeqtor` in this [tutorial](./../tutorial_0001_dataset).

```mermaid
sequenceDiagram
    participant User
    participant Control as Control Sequence
    participant Device as Quantum Device<br/>(Real or Simulator)
    participant Data as ExperimentData
    participant Storage as File System
    
    User->>Control: Define atomic control action
    User->>Control: Create ControlSequence
    Control->>User: Validate & return sequence
    
    alt Real Hardware
        User->>Device: Setup the device
        Note over Device: Physical quantum device<br/>with real noise & decoherence
    else Simulation
        User->>Device: Setup Hamiltonian & Solver
        Note over Device: Local simulation<br/>with modeled noise
    end
    
    loop For each sample (e.g., 100x)
        User->>Control: Sample parameters
        Control->>User: Return control params
        User->>Device: Execute with params
        Device->>User: Return expectation values
        User->>Data: Store row with make_row()
    end
    
    User->>Data: Create ExperimentData
    Data->>Storage: Save to disk
    Storage->>User: Load ExperimentData back
    
    Note over User,Storage: Same data format regardless<br/>of real device or simulator
```

## Characterization Phase

```mermaid
sequenceDiagram
    participant User
    participant ExpData as Experimental Data<br/>(Predefined Model M1)
    participant Model as Neural Network Model
    participant Optimizer as Training Optimizer
    participant Storage as File System
    
    User->>ExpData: Generate synthetic experimental data
    Note over ExpData: Predefined data model M1<br/>100 samples with shot noise<br/>Trotter simulation (10,000 steps)
    ExpData->>User: Return measurement results & whitebox solver
    
    User->>User: Prepare dataset
    Note over User: Split data (90% train, 10% test)<br/>Apply DRAG feature mapping
    
    User->>Model: Create WoModel architecture
    Note over Model: Neural network with<br/>hidden_sizes_1=[10], hidden_sizes_2=[10]
    
    User->>Optimizer: Setup optimizer (5000 epochs)
    
    loop Training Loop (5000 epochs)
        User->>Model: Forward pass with training data
        Model->>User: Return predictions
        User->>User: Calculate MSEE loss
        User->>Optimizer: Update model parameters
        Optimizer->>Model: Apply gradients
        User->>User: Validate on test data
        User->>User: Log training & validation metrics
    end
    
    User->>Storage: Save trained model
    Note over Storage: ModelData with params & config<br/>saved as JSON file
    Storage->>User: Confirm model saved
    
    User->>User: Create predictive model function
    Note over User: Embed whitebox solver for<br/>complete parameter-to-expectation pipeline

```

## Control Calibration Phase

```mermaid
sequenceDiagram
    participant User
    participant PredModel as Predictive Model
    participant CostFn as Cost Function
    participant Calibrator as Parameter Optimizer
    participant QDevice as Quantum Device
    participant Storage as File System
    
    User->>Storage: Load trained model
    Storage->>User: Return ModelData (params & config)
    
    User->>PredModel: Initialize predictive model
    Note over PredModel: Combine trained model with<br/>whitebox solver for full pipeline
    
    User->>User: Define target quantum gate (√X)
    User->>CostFn: Create AGF-based cost function
    Note over CostFn: Minimize (1 - AGF)²<br/>where AGF = Average Gate Fidelity
    
    User->>User: Set parameter bounds from control sequence
    User->>Calibrator: Initialize optimization
    Note over Calibrator: 1000 iterations with<br/>random initial parameters
    
    loop Calibration Loop (1000 iterations)
        User->>PredModel: Predict expectation values for params
        PredModel->>User: Return 18 expectation values
        User->>CostFn: Calculate Average Gate Fidelity
        CostFn->>User: Return infidelity cost & AGF
        User->>Calibrator: Update control parameters
        Calibrator->>User: Return optimized params
        User->>User: Log optimization progress
    end
    
    User->>User: Extract final optimized parameters
    Note over User: Final params: [1.938, 0.051]<br/>Predicted AGF: 0.975
    
    Note over User,QDevice: Benchmarking Phase
    User->>QDevice: Test with optimized parameters
    QDevice->>User: Return actual expectation values
    User->>User: Calculate real AGF: 0.973
    
    Note over User: Validation: Predicted vs Actual<br/>AGF difference < 0.002

```

## Physics
