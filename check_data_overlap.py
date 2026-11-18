
from datasets import load_dataset
from sklearn.model_selection import train_test_split

print("Loading QQP Dataset...")

dataset = load_dataset("nyu-mll/glue", "qqp")

df = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(10000))
    .to_pandas()
)

df = df.dropna(
    subset=["question1", "question2", "label"]
).reset_index(drop=True)

train_df, remaining_df = train_test_split(
    df,
    test_size=0.40,
    random_state=42,
    stratify=df["label"]
)

validation_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    stratify=remaining_df["label"]
)

def get_questions(data):
    return set(
        data["question1"].tolist()
        + data["question2"].tolist()
    )

train_questions = get_questions(train_df)
validation_questions = get_questions(validation_df)
test_questions = get_questions(test_df)

print("\nQuestion Overlap Analysis")
print("------------------------------")

for name, first, second in [
    ("Train vs Validation", train_questions, validation_questions),
    ("Train vs Test", train_questions, test_questions),
    ("Validation vs Test", validation_questions, test_questions)
]:
    overlap = first.intersection(second)

    print(f"\n{name}")
    print("Overlapping Questions:", len(overlap))
    print(
        "Percentage:",
        round(len(overlap) / len(second) * 100, 2),
        "%"
    )

def get_pairs(data):
    return set(
        (q1, q2)
        for q1, q2 in zip(
            data["question1"],
            data["question2"]
        )
    )

train_pairs = get_pairs(train_df)
validation_pairs = get_pairs(validation_df)
test_pairs = get_pairs(test_df)

print("\nExact Question Pair Overlap")
print("------------------------------")
print("Train vs Validation:", len(train_pairs & validation_pairs))
print("Train vs Test:", len(train_pairs & test_pairs))
print("Validation vs Test:", len(validation_pairs & test_pairs))

print("\nOverlap Check Completed!")
