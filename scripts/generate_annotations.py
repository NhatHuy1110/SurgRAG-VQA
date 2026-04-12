import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames_v3"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

BLUEPRINT_FILE = FRAMES_DIR / "question_blueprint.json"
METADATA_FILE = FRAMES_DIR / "frame_metadata.json"

QUESTIONS_OUT = ANNOTATIONS_DIR / "questions_v3.json"
RETRIEVAL_OUT = ANNOTATIONS_DIR / "retrieval_eval_v3.json"


QTYPE_TEMPLATE_BANK = {
    "recognition": [
        "What is the most likely anatomical structure or operative target highlighted in this laparoscopic frame?",
        "Which visible structure is most important to recognize in this frame before proceeding?",
        "What structure is most likely being visualized in the current operative field?",
    ],
    "workflow_phase": [
        "Which phase of laparoscopic cholecystectomy is most likely represented in this frame?",
        "Based on the current view, what operative step is most likely underway?",
        "What is the most likely procedural phase shown in this frame, and what usually comes next?",
    ],
    "anatomy_landmark": [
        "Which anatomical landmark or relationship should be identified in this frame to maintain safe orientation?",
        "What key landmark is most relevant in this frame for orienting dissection safely?",
        "Which anatomy-oriented landmark should the surgeon confirm in this view?",
    ],
    "safety_verification": [
        "Is there enough information in this frame to proceed safely, and what must be verified first?",
        "Before any clipping or division, what safety check is most important in this frame?",
        "Does this frame support safe continuation of dissection, or is additional verification required?",
    ],
    "risk_pitfall": [
        "What is the main operative risk or pitfall suggested by this frame?",
        "What complication risk should the surgeon be most alert to in this frame?",
        "What is the most important danger or misidentification risk in this operative view?",
    ],
}


CLASS_LABELS = {
    "grasper": "a laparoscopic grasper",
    "hepatic_vein": "vascular liver-side anatomy near the hepatic vein region",
    "liver_ligament": "the liver surface or adjacent hepatobiliary tissue",
}


QTYPE_KEYWORDS = {
    "recognition": ["gallbladder", "anatomy", "landmark", "laparoscopic", "structure"],
    "workflow_phase": ["cholecystectomy", "phase", "dissection", "exposure", "workflow"],
    "anatomy_landmark": ["hepatocystic triangle", "landmark", "cystic duct", "cystic artery", "anatomy"],
    "safety_verification": ["critical view of safety", "safe dissection", "cystic duct", "cystic artery", "verification"],
    "risk_pitfall": ["bile duct injury", "misidentification", "bleeding", "bailout", "risk"],
}


QTYPE_MIN_CHUNK = {
    "recognition": "gallbladder",
    "workflow_phase": "cholecystectomy",
    "anatomy_landmark": "hepatocystic",
    "safety_verification": "critical view",
    "risk_pitfall": "bile duct",
}


CLASS_KEYWORDS = {
    "grasper": ["grasper", "traction", "instrument"],
    "hepatic_vein": ["hepatic vein", "liver bed", "vascular"],
    "liver_ligament": ["liver", "gallbladder", "hepatocystic triangle"],
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def class_names(value) -> list[str]:
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, list):
        return list(value)
    return []


def make_question(item: dict) -> str:
    qtype = item["question_type"]
    diff = item["difficulty"]
    classes = class_names(item.get("classes_detected", []))
    idx = int(item["qid"][1:]) - 1
    base = QTYPE_TEMPLATE_BANK[qtype][idx % len(QTYPE_TEMPLATE_BANK[qtype])]

    if qtype == "recognition" and "grasper" in classes:
        return "Which structure and instrument interaction is most likely visible in this frame?"
    if qtype == "workflow_phase" and item["should_defer"]:
        return "Does this frame provide enough visual information to determine the operative phase confidently, or should the system defer?"
    if qtype == "anatomy_landmark" and diff == "hard":
        return "Despite the complex view, which landmark or anatomical relationship is most important to confirm before continuing dissection?"
    if qtype == "safety_verification" and item["should_defer"]:
        return "Should the system defer on this frame because safe progression cannot be verified, and why?"
    if qtype == "risk_pitfall" and item["should_defer"]:
        return "What high-risk pitfall is suggested by this frame, and why would defer be the safer response?"
    return base


def make_gold_answer_stub(item: dict) -> str:
    qtype = item["question_type"]
    diff = item["difficulty"]
    classes = class_names(item.get("classes_detected", []))

    if item["should_defer"]:
        if qtype == "workflow_phase":
            return (
                "DEFER — the visual evidence is not strong enough to assign the operative "
                "phase safely; the frame should be interpreted cautiously and the system "
                "should defer until anatomy or the operative step is clearer."
            )
        if qtype == "safety_verification":
            return (
                "DEFER — safe continuation cannot be verified confidently from this single "
                "frame because the anatomy or dissection status is too uncertain to support "
                "a reliable safety judgment."
            )
        if qtype == "risk_pitfall":
            return (
                "DEFER — this frame suggests a potentially high-risk situation with "
                "ambiguous anatomy or limited visibility, so a conservative defer response "
                "is safer than a definitive action recommendation."
            )
        return (
            "DEFER — the frame does not provide enough reliable visual information for a "
            "safe, frame-specific answer."
        )

    if qtype == "recognition":
        if "grasper" in classes:
            return (
                "The frame most likely shows hepatobiliary tissue being manipulated with a "
                "laparoscopic grasper; the key recognizable elements are the operative field "
                "and the instrument-tissue interaction."
            )
        if "hepatic_vein" in classes and "liver_ligament" in classes:
            return (
                "The frame most likely shows liver-side hepatobiliary anatomy with visible "
                "vascular markings near the gallbladder dissection field."
            )
        return (
            "The frame most likely shows the liver or adjacent hepatobiliary tissue in a "
            "relatively clear laparoscopic view."
        )

    if qtype == "workflow_phase":
        if diff == "easy":
            return (
                "This frame most likely corresponds to a visual exposure or inspection stage "
                "before a critical irreversible step, when the team is orienting to the field."
            )
        if "grasper" in classes:
            return (
                "This frame most likely represents active traction-assisted dissection in the "
                "hepatobiliary field rather than clipping or specimen extraction."
            )
        return (
            "This frame most likely represents active operative dissection or exposure within "
            "the cholecystectomy workflow."
        )

    if qtype == "anatomy_landmark":
        if "hepatic_vein" in classes:
            return (
                "The most relevant landmark is the relationship between the liver-side tissue "
                "and the hepatobiliary dissection plane; orientation should remain close to the "
                "gallbladder side and away from deeper critical structures."
            )
        return (
            "The key task is to confirm a safe hepatobiliary landmark relationship, such as "
            "the gallbladder side of dissection and the expected orientation of the "
            "hepatocystic region, before proceeding."
        )

    if qtype == "safety_verification":
        return (
            "Safe progression requires confirming the intended dissection plane and not "
            "assuming the critical view of safety is achieved from a limited image alone; "
            "clipping or division should only proceed after clear anatomical verification."
        )

    return (
        "The primary pitfall in this frame is misidentification or unsafe continuation in a "
        "hepatobiliary dissection field; the surgeon should stay oriented to the gallbladder "
        "side and remain alert for bile duct or bleeding risk."
    )


def make_notes(item: dict, meta: dict) -> str:
    top_scores = sorted(
        meta["question_type_scores"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    return (
        f"Draft scaffold for manual annotation. "
        f"Assigned={item['question_type']}|{item['difficulty']}; "
        f"should_defer={item['should_defer']}; "
        f"top3_qtypes={top_scores}; "
        f"classes={meta.get('classes_detected', [])}; "
        f"quality={meta.get('quality_score')}"
    )


def make_retrieval_keywords(item: dict) -> list[str]:
    keywords = list(QTYPE_KEYWORDS[item["question_type"]])
    for cls in class_names(item.get("classes_detected", [])):
        keywords.extend(CLASS_KEYWORDS.get(cls, []))

    if item["should_defer"]:
        keywords.extend(["defer", "uncertain anatomy", "unsafe to continue"])
    if item["difficulty"] == "hard":
        keywords.extend(["difficult cholecystectomy", "bailout"])
    elif item["difficulty"] == "medium":
        keywords.extend(["safe dissection", "operative orientation"])
    else:
        keywords.extend(["basic orientation", "clear view"])

    deduped = []
    for kw in keywords:
        if kw not in deduped:
            deduped.append(kw)
    return deduped[:8]


def build_outputs():
    blueprint = load_json(BLUEPRINT_FILE)
    metadata = load_json(METADATA_FILE)
    metadata_by_frame = {item["file_name"]: item for item in metadata}

    questions = []
    retrieval = []

    for item in blueprint:
        meta = metadata_by_frame[item["frame"]]
        question = {
            "qid": item["qid"],
            "frame": item["frame"],
            "question": make_question(item),
            "question_type": item["question_type"],
            "difficulty": item["difficulty"],
            "should_defer": item["should_defer"],
            "gold_answer": make_gold_answer_stub(item),
            "notes": make_notes(item, meta),
            "annotation_status": "ready_run_needs_expert_review",
            "source_frame": meta["source"],
            "video_id": meta["video_id"],
            "frame_index": meta["frame_index"],
            "classes_detected": meta.get("classes_detected", []),
        }
        questions.append(question)

        retrieval.append(
            {
                "qid": item["qid"],
                "question": question["question"],
                "relevant_keywords": make_retrieval_keywords(question),
                "min_acceptable_chunk_contains": QTYPE_MIN_CHUNK[item["question_type"]],
                "question_type": item["question_type"],
                "difficulty": item["difficulty"],
                "should_defer": item["should_defer"],
                "expected_collections": expected_collections(item["question_type"]),
                "annotation_status": "ready_run_needs_expert_review",
            }
        )

    return questions, retrieval


def expected_collections(qtype: str) -> list[str]:
    if qtype == "recognition":
        return ["visual_ontology", "biliary_anatomy_landmarks"]
    if qtype == "workflow_phase":
        return ["safe_chole_guideline", "biliary_anatomy_landmarks"]
    if qtype == "anatomy_landmark":
        return ["biliary_anatomy_landmarks", "visual_ontology"]
    if qtype == "safety_verification":
        return ["safe_chole_guideline", "complication_management"]
    return ["complication_management", "safe_chole_guideline"]


def main():
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    questions, retrieval = build_outputs()

    QUESTIONS_OUT.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    RETRIEVAL_OUT.write_text(
        json.dumps(retrieval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] questions -> {QUESTIONS_OUT}")
    print(f"[DONE] retrieval -> {RETRIEVAL_OUT}")
    print(f"[DONE] total questions: {len(questions)}")


if __name__ == "__main__":
    main()
