from datasets import load_dataset

def main():
    dataset = load_dataset("hellaswag")
    print(dataset)

if __name__ == "__main__":
    main()
