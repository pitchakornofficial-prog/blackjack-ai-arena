import { useId, useState } from "react";
import { Check, X } from "lucide-react";
import { TRAINING_DURATION_OPTIONS } from "./ProjectCreateModal";

type ProjectEditModalProps = {
  gameName: string;
  projectName: string;
  minutes: number;
  onClose: () => void;
  onSave: (name: string, minutes: number) => void;
};

export default function ProjectEditModal({
  gameName,
  projectName,
  minutes,
  onClose,
  onSave
}: ProjectEditModalProps) {
  const nameId = useId();
  const durationId = useId();
  const [name, setName] = useState(projectName);
  const [duration, setDuration] = useState(minutes);

  return (
    <div className="game-modal-overlay project-create-overlay" onMouseDown={event => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <form
        className="game-modal project-create-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${nameId}-title`}
        onSubmit={event => {
          event.preventDefault();
          const normalizedName = name.trim();
          if (!normalizedName) return;
          onSave(normalizedName, duration);
        }}
      >
        <div className="game-modal-header">
          <div>
            <span className="eyebrow">{gameName.toUpperCase()} · PROJECT SETTINGS</span>
            <h2 id={`${nameId}-title`} className="game-modal-title">แก้ไขโปรเจกต์</h2>
          </div>
          <button className="top-icon" type="button" onClick={onClose} aria-label="ปิดหน้าต่าง">
            <X size={15} />
          </button>
        </div>
        <div className="project-modal-body">
          <label className="field-label" htmlFor={nameId}>ชื่อโปรเจกต์</label>
          <input
            id={nameId}
            value={name}
            onChange={event => setName(event.target.value)}
            maxLength={60}
            required
            autoFocus
          />
          <label className="field-label" htmlFor={durationId}>เวลาเป้าหมายต่อรอบการฝึก</label>
          <select
            id={durationId}
            value={duration}
            onChange={event => setDuration(Number(event.target.value))}
          >
            {TRAINING_DURATION_OPTIONS.map(option => (
              <option key={option} value={option}>{option} นาที</option>
            ))}
          </select>
          <p className="field-help">แก้ไขการตั้งค่านี้จะไม่ลบโมเดลหรือข้อมูลการฝึกเดิม</p>
        </div>
        <div className="game-modal-footer">
          <button className="ghost-button" type="button" onClick={onClose}>ยกเลิก</button>
          <button className="primary-action" type="submit">
            <Check size={14} /> บันทึก
          </button>
        </div>
      </form>
    </div>
  );
}
