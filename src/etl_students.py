import os
import pandas as pd

INPUT_XLSX = os.path.join("data", "ML-DATA.xlsx")
OUT_CSV = os.path.join("data", "students_master.csv")

def main():
    # Read all sheets, then concatenate anything that has the expected columns
    xls = pd.ExcelFile(INPUT_XLSX)
    frames = []
    for sh in xls.sheet_names:
        df = xls.parse(sh)
        cols = [c.strip() if isinstance(c, str) else c for c in df.columns]
        df.columns = cols
        needed = {"Class Name", "Register No", "Student Name", "Classes Held", "Classes Attended", "Percentage"}
        if needed.issubset(set(df.columns)):
            frames.append(df[list(needed)].copy())

    if not frames:
        raise ValueError("Could not find expected columns in any sheet.")

    all_df = pd.concat(frames, ignore_index=True)

    # Normalize + rename to your project-style names
    out = all_df.rename(columns={
        "Register No": "student_id",
        "Student Name": "student_name",
        "Class Name": "class_name",
        "Classes Held": "classes_held",
        "Classes Attended": "classes_attended",
        "Percentage": "course_attendance_rate"
    })

    # Basic cleanup
    out["student_id"] = out["student_id"].astype(str).str.strip()
    out["course_attendance_rate"] = pd.to_numeric(out["course_attendance_rate"], errors="coerce")

    # If the same student appears multiple times, keep max attendance (or change to mean)
    out = (out.sort_values("course_attendance_rate", ascending=False)
              .drop_duplicates(subset=["student_id"], keep="first"))

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} rows={len(out)}")

if __name__ == "__main__":
    main()
