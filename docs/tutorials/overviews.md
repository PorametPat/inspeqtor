# Overviews

!!! goal
    Hi! This page is intended for user to understand the overall concept of `inspeqtor` in the high-level first. You might find this page useful when you reviews the interaction between modules and functions offered by `inspeqtor`.

We catergorized characterization and calibration of the quantum device into multiple phase. You might not necessary needs to do or understand every phases and chose to work with specific phases.

```mermaid
sequenceDiagram
    participant User
    participant Model as Predictive Model
    participant Device as Quantum Device
    note over User, Device: Characterization
    loop 
    User ->> Device: Perform experiments
    Device ->> User: Data
    User ->> Model: Characterization
    opt Selection strategy
        Model ->> User: Select new experiments
    end
    end
    note over User, Device: Calibration
    loop Optimization
    User <<->> Model: Find the control that <br/> maximize fidelity
    end
    User ->> Device: Deploy calibrated control
    Note over User, Device: Operational
    loop Operating and monitering
    User ->> Device: Use device
    User ->> Device: Check the quality
    end
```

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
    participant Data
    participant Optimizer
    participant Models

    Note over User,Data: Data Preparation
    User->>Data: Prepare & Split Data
    Data-->>User: Return Training & Testing Data

    Note over User,Models: Model Initialization
    alt Statistical Model (DNN)
        User->>Optimizer: Define Model & Loss Function
        Optimizer->>Models: Initialize Parameters
    else Probabilistic Model (BNN)
        User->>Optimizer: Define Model, Prior, Guide, & SVI
        Optimizer->>Models: Initialize SVI State
    end

    Note over Optimizer,Models: Model Training
    User->>Optimizer: Start Training Loop

    loop For each epoch
        Optimizer->>Models: Update parameters
        Note right of Optimizer: Validates against testing data
    end

    Optimizer-->>User: Return Trained Model

```

## Alternative Characterization phase

```mermaid
sequenceDiagram
    participant User
    participant Strategy as Abstract Strategy
    participant Model
    participant Device

    User->>Strategy: Prepare Experiment

    loop Characterization Loop
        Strategy->>Strategy: Select next experiment parameters
        Note right of Strategy: This can be Random (Open-Loop) or <br> Model-Informed/Adaptive (Closed-Loop).

        Strategy-->>User: Recommend experiment

        User->>Device: Perform experiment
        Device-->>User: Measurement data

        User->>Model: Update/Characterize Model
        
        Model-->>Strategy: Provide Posterior Model (if adaptive)

        Strategy->>Strategy: Check termination condition
    end
    
    Strategy-->>User: Return Final Characterized Model
```

## Control Calibration Phase

```mermaid
sequenceDiagram
    participant User
    participant Optimizer
    participant CostFunction as Cost Function <br/> (e.g., Avg. Gate Infidelity)
    participant Model as PredictiveModel
    participant Device as Quantum Device<br/>(Real or Simulator)
    
    Note over User, Model: Starts with Trained Predictive Model from Characterization
    
    User->>CostFunction: Define(Target Gate, PredictiveModel)
    User->>Optimizer: Start Optimization(CostFunction, Initial Params)
    
    loop Optimization Steps
        Optimizer->>CostFunction: Evaluate(current_params)
        CostFunction->>Model: Predict(current_params)
        Model-->>CostFunction: Return Expectation Values
        CostFunction-->>Optimizer: Return Loss (Infidelity)
        Optimizer->>Optimizer: Update Parameters
    end
    
    Optimizer-->>User: Return Optimized Control Parameters
    
    %% alt Benchmarking
    User->>Device: Execute(Optimized Params)
    Device-->>User: Return Measured Fidelity
    User->>Model: Predict(Optimized Params)
    Model-->>User: Return Predicted Fidelity
```

## Physics
