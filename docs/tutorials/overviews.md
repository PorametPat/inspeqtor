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

## Control Calibration Phase

## Physics
