import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const API_BASE = "http://localhost:8005";

interface DatasetInfo {
  name: string;
  path: string;
  image_count: number;
  size_mb: number;
  has_images: boolean;
  selected?: boolean;
}

interface PhaseConfig {
  phase_number: number;
  name: string;
  runs_per_project: number;
  passes: number;
  strategy_override?: string;
  preset_override?: string;
  update_model: boolean;
  context_jitter: boolean;
  context_jitter_percent: number;
  shuffle_order: boolean;
  session_execution_mode: string;
}


export default function TrainingPipelinePage() {
  const navigate = useNavigate();

  // Step 1: Dataset Selection
  const [baseDirectory, setBaseDirectory] = useState("");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [scanning, setScanning] = useState(false);

  // Step 2: Shared Configuration
  const [aiInputMode, setAiInputMode] = useState("exif_plus_flight_plan");
  const [aiSelectorStrategy, setAiSelectorStrategy] = useState("contextual_continuous");
  const [maxSteps, setMaxSteps] = useState(5000);
  const [evalInterval, setEvalInterval] = useState(1000);
  const [logInterval, setLogInterval] = useState(100);
  const [densifyUntil, setDensifyUntil] = useState(4000);
  const [imagesMaxSize, setImagesMaxSize] = useState(1600);

  // Step 3: Training Schedule
  const [phases, setPhases] = useState<PhaseConfig[]>([
    {
      phase_number: 1,
      name: "Baseline Collection",
      runs_per_project: 1,
      passes: 1,
      strategy_override: "preset_bias",
      preset_override: "balanced",
      update_model: false,
      context_jitter: false,
      context_jitter_percent: 0,
      shuffle_order: false,
      session_execution_mode: "test",
    },
    {
      phase_number: 2,
      name: "Initial Exploration",
      runs_per_project: 1,
      passes: 1,
      update_model: true,
      context_jitter: false,
      context_jitter_percent: 0,
      shuffle_order: true,
      session_execution_mode: "train",
    },
    {
      phase_number: 3,
      name: "Multi-Pass Learning",
      runs_per_project: 1,
      passes: 5,
      update_model: true,
      context_jitter: true,
      context_jitter_percent: 5,
      shuffle_order: true,
      session_execution_mode: "train",
    },
  ]);

  // Step 4: Thermal Management
  const [thermalEnabled, setThermalEnabled] = useState(true);
  const [thermalStrategy, setThermalStrategy] = useState("fixed_interval");
  const [cooldownMinutes, setCooldownMinutes] = useState(10);

  // Step 5: Review
  const [pipelineName, setPipelineName] = useState(`training_${new Date().toISOString().split("T")[0]}`);
  const [creating, setCreating] = useState(false);

  // UI State
  const [currentStep, setCurrentStep] = useState(1);

  // Scan directory for datasets
  const handleScanDirectory = async () => {
    if (!baseDirectory.trim()) {
      alert("Please enter a directory path");
      return;
    }

    setScanning(true);
    try {
      const response = await axios.post(`${API_BASE}/training-pipeline/scan-directory`, {
        base_directory: baseDirectory,
      });

      const scannedDatasets = response.data.datasets.map((d: DatasetInfo) => ({
        ...d,
        selected: true, // Auto-select all by default
      }));

      setDatasets(scannedDatasets);
    } catch (error: any) {
      console.error("Failed to scan directory:", error);
      alert(`Failed to scan directory: ${error.response?.data?.detail || error.message}`);
    } finally {
      setScanning(false);
    }
  };

  // Calculate total runs
  const calculateTotalRuns = () => {
    const selectedCount = datasets.filter((d) => d.selected).length;
    let total = 0;
    for (const phase of phases) {
      total += phase.runs_per_project * phase.passes * selectedCount;
    }
    return total;
  };

  // Calculate estimated time
  const calculateEstimatedTime = () => {
    const totalRuns = calculateTotalRuns();
    const trainingMinutes = totalRuns * 8; // Assume 8 minutes per run
    const cooldownTime = thermalEnabled ? totalRuns * cooldownMinutes : 0;
    const totalMinutes = trainingMinutes + cooldownTime;
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return { hours, minutes, totalMinutes };
  };

  // Create pipeline
  const handleCreatePipeline = async () => {
    const selectedDatasets = datasets.filter((d) => d.selected);

    if (selectedDatasets.length === 0) {
      alert("Please select at least one dataset");
      return;
    }

    setCreating(true);
    try {
      // Build configuration
      const config = {
        name: pipelineName,
        base_directory: baseDirectory,
        projects: selectedDatasets.map((d) => ({
          name: d.name,
          dataset_path: d.path,
          image_count: d.image_count,
          created: false,
        })),
        shared_config: {
          ai_input_mode: aiInputMode,
          ai_selector_strategy: aiSelectorStrategy,
          max_steps: maxSteps,
          eval_interval: evalInterval,
          log_interval: logInterval,
          densify_until_iter: densifyUntil,
          images_max_size: imagesMaxSize,
        },
        phases: phases,
        thermal_management: {
          enabled: thermalEnabled,
          strategy: thermalStrategy,
          cooldown_minutes: cooldownMinutes,
          gpu_temp_threshold: 70,
          check_interval_seconds: 30,
          max_wait_minutes: 30,
        },
        failure_handling: {
          continue_on_failure: true,
          max_retries_per_run: 1,
          skip_project_after_failures: 3,
        },
      };

      // Create pipeline
      const response = await axios.post(`${API_BASE}/training-pipeline/create`, config);

      alert(`Pipeline created successfully! ID: ${response.data.id}`);
      navigate("/"); // Return to dashboard

    } catch (error: any) {
      console.error("Failed to create pipeline:", error);
      alert(`Failed to create pipeline: ${error.response?.data?.detail || error.message}`);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <h1>Training Pipeline</h1>
        <button onClick={() => navigate("/")} style={{ padding: "8px 16px" }}>
          Back to Dashboard
        </button>
      </div>

      {/* Progress Indicator */}
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "30px", padding: "10px", background: "#f5f5f5", borderRadius: "4px" }}>
        {[1, 2, 3, 4, 5].map((step) => (
          <div key={step} style={{ flex: 1, textAlign: "center" }}>
            <div
              style={{
                width: "40px",
                height: "40px",
                borderRadius: "50%",
                background: currentStep >= step ? "#4CAF50" : "#ddd",
                color: "white",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: "bold",
                marginBottom: "5px",
              }}
            >
              {step}
            </div>
            <div style={{ fontSize: "12px", color: currentStep >= step ? "#333" : "#999" }}>
              {step === 1 && "Datasets"}
              {step === 2 && "Config"}
              {step === 3 && "Schedule"}
              {step === 4 && "Thermal"}
              {step === 5 && "Review"}
            </div>
          </div>
        ))}
      </div>

      {/* Step 1: Dataset Selection */}
      {currentStep === 1 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 1: Dataset Selection</h2>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>Base Directory:</label>
            <div style={{ display: "flex", gap: "10px" }}>
              <input
                type="text"
                value={baseDirectory}
                onChange={(e) => setBaseDirectory(e.target.value)}
                placeholder="E:/Thesis/exp_new_method"
                style={{ flex: 1, padding: "8px" }}
              />
              <button onClick={handleScanDirectory} disabled={scanning} style={{ padding: "8px 16px" }}>
                {scanning ? "Scanning..." : "Scan Directory"}
              </button>
            </div>
          </div>

          {datasets.length > 0 && (
            <div>
              <h3>Discovered Datasets ({datasets.filter((d) => d.selected).length}/{datasets.length} selected):</h3>

              <div style={{ marginBottom: "10px" }}>
                <button onClick={() => setDatasets(datasets.map((d) => ({ ...d, selected: true })))} style={{ marginRight: "10px" }}>
                  Select All
                </button>
                <button onClick={() => setDatasets(datasets.map((d) => ({ ...d, selected: false })))}>
                  Deselect All
                </button>
              </div>

              <div style={{ maxHeight: "300px", overflowY: "auto", border: "1px solid #ddd", padding: "10px" }}>
                {datasets.map((dataset, idx) => (
                  <div key={idx} style={{ padding: "8px", borderBottom: "1px solid #eee", display: "flex", alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={dataset.selected}
                      onChange={(e) => {
                        const updated = [...datasets];
                        updated[idx].selected = e.target.checked;
                        setDatasets(updated);
                      }}
                      style={{ marginRight: "10px" }}
                    />
                    <div style={{ flex: 1 }}>
                      <strong>{dataset.name}</strong>
                      <div style={{ fontSize: "12px", color: "#666" }}>
                        Images: {dataset.image_count} | Size: {dataset.size_mb.toFixed(1)} MB
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: "20px", textAlign: "right" }}>
            <button onClick={() => setCurrentStep(2)} disabled={datasets.filter((d) => d.selected).length === 0} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Shared Configuration */}
      {currentStep === 2 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 2: Shared Training Configuration</h2>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>AI Input Mode:</label>
            <select value={aiInputMode} onChange={(e) => setAiInputMode(e.target.value)} style={{ width: "100%", padding: "8px" }}>
              <option value="exif_only">EXIF Only</option>
              <option value="exif_plus_flight_plan">EXIF + Flight Plan</option>
              <option value="exif_plus_flight_plan_plus_external">EXIF + Flight Plan + External</option>
            </select>
          </div>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>Selector Strategy:</label>
            <select value={aiSelectorStrategy} onChange={(e) => setAiSelectorStrategy(e.target.value)} style={{ width: "100%", padding: "8px" }}>
              <option value="contextual_continuous">Contextual Continuous (NEW)</option>
              <option value="continuous_bandit_linear">Continuous Bandit</option>
              <option value="preset_bias">Preset Bias</option>
            </select>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px" }}>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Max Steps:</label>
              <input type="number" value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Eval Interval:</label>
              <input type="number" value={evalInterval} onChange={(e) => setEvalInterval(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Log Interval:</label>
              <input type="number" value={logInterval} onChange={(e) => setLogInterval(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Densify Until:</label>
              <input type="number" value={densifyUntil} onChange={(e) => setDensifyUntil(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "5px" }}>Images Max Size:</label>
              <input type="number" value={imagesMaxSize} onChange={(e) => setImagesMaxSize(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
            </div>
          </div>

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(1)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button onClick={() => setCurrentStep(3)} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Training Schedule */}
      {currentStep === 3 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 3: Training Schedule</h2>

          {phases.map((phase, idx) => (
            <div key={idx} style={{ marginBottom: "20px", padding: "15px", background: "#f9f9f9", borderRadius: "4px" }}>
              <h3>Phase {phase.phase_number}: {phase.name}</h3>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                <div>
                  <label style={{ display: "block", fontSize: "12px", marginBottom: "3px" }}>Runs per project:</label>
                  <input
                    type="number"
                    value={phase.runs_per_project}
                    onChange={(e) => {
                      const updated = [...phases];
                      updated[idx].runs_per_project = Number(e.target.value);
                      setPhases(updated);
                    }}
                    style={{ width: "100%", padding: "6px" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: "12px", marginBottom: "3px" }}>Passes:</label>
                  <input
                    type="number"
                    value={phase.passes}
                    onChange={(e) => {
                      const updated = [...phases];
                      updated[idx].passes = Number(e.target.value);
                      setPhases(updated);
                    }}
                    style={{ width: "100%", padding: "6px" }}
                  />
                </div>
              </div>

              <div style={{ marginTop: "10px", fontSize: "13px", color: "#666" }}>
                Total runs: {phase.runs_per_project * phase.passes * datasets.filter((d) => d.selected).length}
              </div>
            </div>
          ))}

          <div style={{ padding: "15px", background: "#e3f2fd", borderRadius: "4px" }}>
            <strong>Grand Total: {calculateTotalRuns()} runs</strong>
          </div>

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(2)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button onClick={() => setCurrentStep(4)} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Thermal Management */}
      {currentStep === 4 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 4: Thermal Management</h2>

          <div style={{ marginBottom: "15px" }}>
            <label>
              <input type="checkbox" checked={thermalEnabled} onChange={(e) => setThermalEnabled(e.target.checked)} style={{ marginRight: "8px" }} />
              Enable cooldown periods between runs
            </label>
          </div>

          {thermalEnabled && (
            <div>
              <div style={{ marginBottom: "15px" }}>
                <label style={{ display: "block", marginBottom: "5px" }}>Cooldown Strategy:</label>
                <select value={thermalStrategy} onChange={(e) => setThermalStrategy(e.target.value)} style={{ width: "100%", padding: "8px" }}>
                  <option value="fixed_interval">Fixed Interval</option>
                  <option value="temperature_based">Temperature-based (requires GPU monitoring)</option>
                  <option value="time_scheduled">Time-of-day scheduling</option>
                </select>
              </div>

              {thermalStrategy === "fixed_interval" && (
                <div style={{ marginBottom: "15px" }}>
                  <label style={{ display: "block", marginBottom: "5px" }}>Wait time (minutes):</label>
                  <input
                    type="number"
                    value={cooldownMinutes}
                    onChange={(e) => setCooldownMinutes(Number(e.target.value))}
                    style={{ width: "100%", padding: "8px" }}
                  />
                </div>
              )}

              <div style={{ padding: "15px", background: "#fff3cd", borderRadius: "4px" }}>
                <h4>Estimated Total Time:</h4>
                <div>Training time: {calculateTotalRuns()} runs × 8 min ≈ {Math.floor((calculateTotalRuns() * 8) / 60)} hours</div>
                <div>Cooldown time: {calculateTotalRuns()} × {cooldownMinutes} min ≈ {Math.floor((calculateTotalRuns() * cooldownMinutes) / 60)} hours</div>
                <div style={{ fontWeight: "bold", marginTop: "10px" }}>
                  Total: ~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m (~{(calculateEstimatedTime().totalMinutes / 1440).toFixed(1)} days)
                </div>
              </div>
            </div>
          )}

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(3)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button onClick={() => setCurrentStep(5)} style={{ padding: "8px 24px" }}>
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 5: Review & Launch */}
      {currentStep === 5 && (
        <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
          <h2>Step 5: Review & Launch</h2>

          <div style={{ padding: "15px", background: "#f5f5f5", borderRadius: "4px", marginBottom: "15px" }}>
            <h3>Pipeline Summary:</h3>
            <ul>
              <li>Projects: {datasets.filter((d) => d.selected).length}</li>
              <li>Total runs: {calculateTotalRuns()}</li>
              <li>Strategy: {aiSelectorStrategy}</li>
              <li>Estimated duration: ~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</li>
            </ul>
          </div>

          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>Pipeline Name:</label>
            <input
              type="text"
              value={pipelineName}
              onChange={(e) => setPipelineName(e.target.value)}
              style={{ width: "100%", padding: "8px" }}
            />
          </div>

          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
            <button onClick={() => setCurrentStep(4)} style={{ padding: "8px 24px" }}>
              Back
            </button>
            <button
              onClick={handleCreatePipeline}
              disabled={creating}
              style={{ padding: "8px 24px", background: "#4CAF50", color: "white", fontWeight: "bold", border: "none", borderRadius: "4px" }}
            >
              {creating ? "Creating..." : "▶ Start Pipeline"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
