import { useEffect, useId, useRef } from "react";
import type { FormEvent, ReactNode } from "react";
import { createPortal } from "react-dom";
import { Plus, X } from "lucide-react";

export const TRAINING_DURATION_OPTIONS = [1, 5, 10, 15, 30, 60] as const;
export const TRAINING_BATCH_OPTIONS = [50, 100, 200, 500] as const;

type ProjectCreateModalProps = {
  gameName: string;
  onClose: () => void;
  onCreate: () => void;
  children: ReactNode;
};

export default function ProjectCreateModal({
  gameName,
  onClose,
  onCreate,
  children
}: ProjectCreateModalProps) {
  const titleId = useId();
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCloseRef.current();
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onCreate();
  }

  return createPortal(
    <div className="game-modal-overlay project-create-overlay" onMouseDown={event => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <form
        className="game-modal project-create-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onSubmit={handleSubmit}
      >
        <div className="game-modal-header">
          <div>
            <span className="eyebrow">{gameName.toUpperCase()} · NEW PROJECT</span>
            <h2 id={titleId} className="game-modal-title">สร้างโปรเจกต์ใหม่</h2>
          </div>
          <button className="top-icon" type="button" onClick={onClose} aria-label="ปิดหน้าต่าง">
            <X size={15} />
          </button>
        </div>
        <div className="project-modal-body">{children}</div>
        <div className="game-modal-footer">
          <button className="ghost-button" type="button" onClick={onClose}>ยกเลิก</button>
          <button className="primary-action" type="submit">
            <Plus size={14} /> สร้างโปรเจกต์
          </button>
        </div>
      </form>
    </div>,
    document.body
  );
}
