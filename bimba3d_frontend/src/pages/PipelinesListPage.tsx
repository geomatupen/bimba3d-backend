import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Play, Pause, Square, RefreshCw, Trash2, Plus, MoreVertical, Workflow } from "lucide-react";
import { api } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";

interface Pipeline {
  id: string;
  name: string;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  current_phase: number;
  current_pass: number;
  current_project_index: number;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  mean_reward: number | null;
  success_rate: number | null;
  best_reward: number | null;
  last_error: string | null;
  cooldown_active: boolean;
  next_run_scheduled_at: string | null;
}

export default function PipelinesListPage() {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [actioningId, setActioningId] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<{ open: boolean; pipelineId: string | null; pipelineName: string }>({
    open: false,
    pipelineId: null,
    pipelineName: "",
  });
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const navigate = useNavigate();

  const showToast = (message: string, type: "success" | "error" = "success") => {
    setToast({ message, type });
    window.setTimeout(() => setToast(null), 3000);
  };

  const loadPipelines = useCallback(async () => {
    try {
      const res = await api.get("/training-pipeline/list?limit=50");
      setPipelines(res.data.pipelines || []);
    } catch (err) {
      console.error("Failed to load pipelines", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadPipelines();
    const timer = setInterval(loadPipelines, 5000);
    return () => clearInterval(timer);
  }, [loadPipelines]);

  useEffect(() => {
    if (!menuOpenId) return;

    const closeMenu = () => setMenuOpenId(null);
    window.addEventListener("click", closeMenu);
    return () => window.removeEventListener("click", closeMenu);
  }, [menuOpenId]);

  const handleAction = async (pipelineId: string, action: "start" | "pause" | "resume" | "stop", event: React.MouseEvent) => {
    event.stopPropagation();
    if (actioningId) return;
    setActioningId(pipelineId);
    try {
      await api.post(`/training-pipeline/${pipelineId}/${action}`);
      showToast(`Pipeline ${action}ed successfully`, "success");
      await loadPipelines();
    } catch (err: any) {
      showToast(err.response?.data?.detail || `Failed to ${action} pipeline`, "error");
    } finally {
      setActioningId(null);
    }
  };

  const handleDelete = (pipelineId: string, pipelineName: string) => {
    setDeleteConfirm({ open: true, pipelineId, pipelineName });
  };

  const confirmDelete = async () => {
    if (!deleteConfirm.pipelineId) return;

    const pipelineId = deleteConfirm.pipelineId;
    setActioningId(pipelineId);

    try {
      await api.delete(`/training-pipeline/${pipelineId}`);
      showToast("Pipeline deleted successfully", "success");
      await loadPipelines();
    } catch (err: any) {
      showToast(err.response?.data?.detail || "Failed to delete pipeline", "error");
    } finally {
      setActioningId(null);
      setDeleteConfirm({ open: false, pipelineId: null, pipelineName: "" });
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "running":
        return "bg-green-50 text-green-700 border-green-300";
      case "paused":
        return "bg-yellow-50 text-yellow-700 border-yellow-300";
      case "completed":
        return "bg-blue-50 text-blue-700 border-blue-300";
      case "failed":
        return "bg-red-50 text-red-700 border-red-300";
      case "stopped":
        return "bg-gray-50 text-gray-700 border-gray-300";
      default:
        return "bg-slate-50 text-slate-700 border-slate-300";
    }
  };

  const formatDuration = (start: string, end: string | null) => {
    const startTime = new Date(start).getTime();
    const endTime = end ? new Date(end).getTime() : Date.now();
    const diff = endTime - startTime;
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Pipelines</h1>
          <p className="text-sm text-gray-600 mt-0.5">Manage and monitor your automated training pipelines</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => { setRefreshing(true); loadPipelines(); }}
            disabled={refreshing}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            onClick={() => navigate("/training-pipeline")}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            <Plus className="w-3.5 h-3.5" />
            New Pipeline
          </button>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-lg shadow-lg ${
            toast.type === "success" ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
          }`}
        >
          {toast.message}
        </div>
      )}

      {/* Pipeline Cards */}
      {pipelines.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <p className="text-gray-500 text-base mb-4">No pipelines created yet</p>
          <button
            onClick={() => navigate("/training-pipeline")}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
          >
            <Plus className="w-4 h-4" />
            Create Your First Pipeline
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {pipelines.map((pipeline) => (
            <div
              key={pipeline.id}
              className="group relative block rounded-xl border border-slate-300 bg-white hover:shadow-lg transition-all duration-300 shadow-sm overflow-hidden hover:border-blue-400 cursor-pointer"
              onClick={() => navigate(`/pipelines/${pipeline.id}`)}
            >
              <div className="flex items-center gap-3 p-3">
                {/* Icon */}
                <div className="flex-shrink-0 h-12 w-12 rounded-lg bg-gradient-to-br from-indigo-500 to-indigo-600 flex items-center justify-center shadow-md group-hover:scale-105 transition-transform duration-300">
                  <Workflow className="w-6 h-6 text-white" />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-bold text-slate-900 group-hover:text-blue-600 transition-colors mb-0.5 truncate">
                        {pipeline.name}
                      </h3>
                      <p className="text-xs text-slate-500">
                        {pipeline.started_at && (
                          <>
                            {pipeline.completed_at
                              ? `Completed • ${formatDuration(pipeline.started_at, pipeline.completed_at)}`
                              : `Running • ${formatDuration(pipeline.started_at, null)}`
                            }
                          </>
                        )}
                        {!pipeline.started_at && `Created ${new Date(pipeline.created_at).toLocaleDateString()}`}
                      </p>
                    </div>
                    <div className="flex items-center gap-1">
                      <span
                        className={`flex-shrink-0 px-2 py-0.5 rounded-full text-xs font-semibold border ${getStatusColor(
                          pipeline.status
                        )}`}
                      >
                        {pipeline.status}
                      </span>
                      <div className="relative">
                        <button
                          className="p-1 rounded-md hover:bg-slate-100 text-slate-500 hover:text-slate-700"
                          onClick={(e) => {
                            e.stopPropagation();
                            setMenuOpenId((prev) => (prev === pipeline.id ? null : pipeline.id));
                          }}
                          aria-label="Pipeline actions"
                        >
                          <MoreVertical className="w-3.5 h-3.5" />
                        </button>
                        {menuOpenId === pipeline.id && (
                          <div className="absolute right-0 mt-1 w-32 rounded-lg border border-slate-200 bg-white shadow-lg z-20">
                            {pipeline.status === "pending" && (
                              <button
                                className="w-full text-left px-3 py-1.5 text-xs hover:bg-slate-50 flex items-center gap-1.5"
                                onClick={(e) => handleAction(pipeline.id, "start", e)}
                                disabled={actioningId === pipeline.id}
                              >
                                <Play className="w-3 h-3" />
                                Start
                              </button>
                            )}
                            {pipeline.status === "running" && (
                              <>
                                <button
                                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-slate-50 flex items-center gap-1.5"
                                  onClick={(e) => handleAction(pipeline.id, "pause", e)}
                                  disabled={actioningId === pipeline.id}
                                >
                                  <Pause className="w-3 h-3" />
                                  Pause
                                </button>
                                <button
                                  className="w-full text-left px-3 py-1.5 text-xs text-red-600 hover:bg-red-50 flex items-center gap-1.5"
                                  onClick={(e) => handleAction(pipeline.id, "stop", e)}
                                  disabled={actioningId === pipeline.id}
                                >
                                  <Square className="w-3 h-3" />
                                  Stop
                                </button>
                              </>
                            )}
                            {(pipeline.status === "paused" || pipeline.status === "stopped") && (
                              <button
                                className="w-full text-left px-3 py-1.5 text-xs hover:bg-slate-50 flex items-center gap-1.5"
                                onClick={(e) => handleAction(pipeline.id, "resume", e)}
                                disabled={actioningId === pipeline.id}
                              >
                                <Play className="w-3 h-3" />
                                Resume
                              </button>
                            )}
                            {(pipeline.status === "completed" || pipeline.status === "stopped" || pipeline.status === "failed") && (
                              <button
                                className="w-full text-left px-3 py-1.5 text-xs text-red-600 hover:bg-red-50 flex items-center gap-1.5"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setMenuOpenId(null);
                                  handleDelete(pipeline.id, pipeline.name);
                                }}
                                disabled={actioningId === pipeline.id}
                              >
                                <Trash2 className="w-3 h-3" />
                                Delete
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="space-y-0.5">
                    <div className="flex justify-between text-xs font-medium text-slate-600">
                      <span>Progress: {pipeline.completed_runs}/{pipeline.total_runs}</span>
                      <span className="text-blue-600">{Math.round((pipeline.completed_runs / pipeline.total_runs) * 100)}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden shadow-inner">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-indigo-600 transition-all duration-300 rounded-full"
                        style={{ width: `${(pipeline.completed_runs / pipeline.total_runs) * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Stats Row */}
                  <div className="flex items-center gap-4 text-xs">
                    {pipeline.status === "running" && (
                      <span className="text-slate-600">
                        Phase {pipeline.current_phase} • Pass {pipeline.current_pass}
                      </span>
                    )}
                    {pipeline.mean_reward !== null && (
                      <span className="text-slate-600">
                        <span className="text-slate-500">Reward:</span> <span className="font-semibold">{pipeline.mean_reward.toFixed(3)}</span>
                      </span>
                    )}
                    {pipeline.success_rate !== null && (
                      <span className="text-slate-600">
                        <span className="text-slate-500">Success:</span> <span className="font-semibold">{pipeline.success_rate.toFixed(1)}%</span>
                      </span>
                    )}
                    {pipeline.failed_runs > 0 && (
                      <span className="text-red-600 font-semibold">
                        {pipeline.failed_runs} failed
                      </span>
                    )}
                  </div>

                  {/* Error Message */}
                  {pipeline.last_error && (
                    <div className="text-xs text-red-700 bg-red-50 px-2 py-1 rounded border border-red-200 truncate" title={pipeline.last_error}>
                      <strong>Error:</strong> {pipeline.last_error}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmModal
        open={deleteConfirm.open}
        title="Delete Pipeline"
        message={
          <>
            Are you sure you want to delete <strong>{deleteConfirm.pipelineName}</strong>?
            <br />
            <br />
            This will permanently delete:
            <ul className="list-disc ml-5 mt-2">
              <li>All project outputs and COLMAP data</li>
              <li>All training runs and results</li>
              <li>Shared AI models</li>
              <li>The entire pipeline folder</li>
            </ul>
            <br />
            <strong>This action cannot be undone.</strong>
          </>
        }
        confirmLabel="Delete Pipeline"
        cancelLabel="Cancel"
        tone="danger"
        busy={actioningId === deleteConfirm.pipelineId}
        onConfirm={confirmDelete}
        onCancel={() => setDeleteConfirm({ open: false, pipelineId: null, pipelineName: "" })}
      />
    </div>
  );
}
