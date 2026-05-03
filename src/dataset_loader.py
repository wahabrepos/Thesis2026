## dataset_loader.py
import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    DatasetLoader loads and preprocesses the MedQA and PubMedQA datasets based on the provided configuration.
    It supports JSON and CSV file formats and applies dataset-specific filtering and sampling.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DatasetLoader with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
                Expected keys:
                    - experiment: { datasets: { medqa: <path>, pubmedqa: <path> }, sample_size: <int> }
        """
        # Retrieve dataset file paths.
        # config.yaml stores each dataset as a nested dict {train_path, test_path};
        # fall back to treating the value as a plain string for backwards compatibility.
        experiment_config = config.get("experiment", {})
        datasets_config = experiment_config.get("datasets", {})

        medqa_cfg = datasets_config.get("medqa", {})
        self.medqa_path: str = (
            medqa_cfg.get("test_path", "data/datasets/medqa_test.json")
            if isinstance(medqa_cfg, dict) else str(medqa_cfg)
        )

        pubmedqa_cfg = datasets_config.get("pubmedqa", {})
        self.pubmedqa_path: str = (
            pubmedqa_cfg.get("test_path", "data/datasets/pubmedqa_test.json")
            if isinstance(pubmedqa_cfg, dict) else str(pubmedqa_cfg)
        )

        # Retrieve sample size with a default of 1000 if not specified
        self.sample_size: int = experiment_config.get("sample_size", 1000)

        logger.info(f"MedQA dataset path: {self.medqa_path}")
        logger.info(f"PubMedQA dataset path: {self.pubmedqa_path}")
        logger.info(f"Sample size per dataset: {self.sample_size}")

    def _file_exists(self, file_path: str) -> None:
        """
        Checks if the given file exists.
        
        Args:
            file_path (str): Path to the file.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at path '{file_path}' was not found.")

    def _load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Loads a file (JSON or CSV) into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the data file.
        
        Returns:
            pd.DataFrame: DataFrame containing the dataset.
        """
        self._file_exists(file_path)
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext == ".json":
                # Try to read as json and convert to DataFrame
                try:
                    df = pd.read_json(file_path)
                except ValueError:
                    # If pandas fails, use json.load then convert
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format '{ext}' for file {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading file '{file_path}': {e}")
        logger.info(f"Loaded {len(df)} entries from {file_path}")
        return df

    def _preprocess_medqa(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Preprocesses the MedQA dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing MedQA data.
        
        Returns:
            List[Dict[str, Any]]: Preprocessed list of MedQA question dictionaries.
            Each dictionary contains keys: 'question', 'options', 'answer'.
        """
        # Handle nested 'data' field format (e.g., the MedQA JSON has a top-level 'data' dict)
        if 'data' in df.columns and df['data'].apply(lambda x: isinstance(x, dict)).all():
            df = pd.json_normalize(df['data'])

        # Support multiple possible column names for the question and answer fields
        if 'Question' in df.columns and 'question' not in df.columns:
            df = df.rename(columns={'Question': 'question'})
        if 'Correct Answer' in df.columns and 'answer' not in df.columns:
            df = df.rename(columns={'Correct Answer': 'answer'})
        if 'Options' in df.columns and 'options' not in df.columns:
            df = df.rename(columns={'Options': 'options'})

        # Assume MedQA dataset has at least the following columns: 'question', 'options', 'answer'
        required_columns = ['question', 'answer']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"MedQA dataset is missing required column '{col}'")

        # If 'options' column is not present, create an empty list for each record.
        if 'options' not in df.columns:
            df['options'] = [[] for _ in range(len(df))]

        # Sample if dataset is larger than sample_size
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            logger.info(f"MedQA dataset sampled to {self.sample_size} entries.")

        # Convert DataFrame rows to list of dictionaries.
        # The question sent to the pipeline is pre-formatted with labelled options so
        # the generator can output a single letter (A/B/C/D).  The ground-truth stored
        # as "answer" is the option letter — exact letter matching in Evaluation is
        # unambiguous and immune to paraphrase differences.
        option_labels = "ABCDE"
        medqa_data = []
        for _, row in df.iterrows():
            question_text = str(row["question"]).strip()
            raw_options = row["options"] if isinstance(row["options"], list) else str(row["options"]).split(";")
            options = [str(o).strip() for o in raw_options if str(o).strip()]

            # Prefer pre-computed answer_letter (from download script); fall back to
            # matching the answer text against the option list.
            if "answer_letter" in df.columns and str(row["answer_letter"]).strip().upper() in option_labels:
                correct_letter = str(row["answer_letter"]).strip().upper()
            else:
                answer_text = str(row["answer"]).strip().lower()
                correct_letter = ""
                for i, opt in enumerate(options):
                    if opt.lower() == answer_text and i < len(option_labels):
                        correct_letter = option_labels[i]
                        break

            # Build multi-choice prompt appended to the question
            if options:
                choices_block = "\n".join(
                    f"{option_labels[i]}. {opt}"
                    for i, opt in enumerate(options)
                    if i < len(option_labels)
                )
                formatted_question = (
                    f"{question_text}\n\n"
                    f"Answer choices:\n{choices_block}\n\n"
                    f"Output ONLY the single letter (A/B/C/D) of the correct answer."
                )
            else:
                formatted_question = question_text

            entry = {
                "question":      formatted_question,
                "options":       options,
                "answer":        correct_letter or str(row["answer"]).strip(),
                "answer_text":   str(row["answer"]).strip(),
                "dataset_type":  "medqa",
            }
            medqa_data.append(entry)

        logger.info(f"Preprocessed MedQA dataset with {len(medqa_data)} entries.")
        return medqa_data

    def _preprocess_pubmedqa(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Preprocesses the PubMedQA dataset by filtering out rows with 'maybe' as answer.
        
        Args:
            df (pd.DataFrame): DataFrame containing PubMedQA data.
        
        Returns:
            List[Dict[str, Any]]: Preprocessed list of PubMedQA question dictionaries.
            Each dictionary contains keys: 'question', 'answer'.
        """
        # Assume PubMedQA dataset has at least the following columns: 'question', 'final_decision'
        required_columns = ['question', 'final_decision']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"PubMedQA dataset is missing required column '{col}'")

        # Filter out rows where the answer is 'maybe'
        df_filtered = df[~df["final_decision"].astype(str).str.lower().str.strip().eq("maybe")]
        logger.info(f"Filtered PubMedQA dataset: {len(df)} before, {len(df_filtered)} after removing 'maybe' answers.")

        # Sample if dataset is larger than sample_size
        if len(df_filtered) > self.sample_size:
            df_filtered = df_filtered.sample(n=self.sample_size, random_state=42)
            logger.info(f"PubMedQA dataset sampled to {self.sample_size} entries.")

        # Convert DataFrame rows to list of dictionaries
        pubmedqa_data = []
        for _, row in df_filtered.iterrows():
            entry = {
                "question": str(row["question"]).strip(),
                "answer": str(row["final_decision"]).strip()
            }
            # If additional fields such as 'abstract' exist, include them.
            if "abstract" in df_filtered.columns:
                entry["abstract"] = str(row["abstract"]).strip()
            pubmedqa_data.append(entry)
        logger.info(f"Preprocessed PubMedQA dataset with {len(pubmedqa_data)} entries.")
        return pubmedqa_data

    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads and preprocesses both MedQA and PubMedQA datasets.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with keys 'medqa' and 'pubmedqa' containing the preprocessed data.
        """
        datasets: Dict[str, List[Dict[str, Any]]] = {}

        # Process MedQA dataset
        try:
            medqa_df = self._load_file_to_dataframe(self.medqa_path)
            medqa_processed = self._preprocess_medqa(medqa_df)
            datasets["medqa"] = medqa_processed
        except Exception as e:
            logger.error(f"Failed to load and preprocess MedQA dataset: {e}")
            raise

        # Process PubMedQA dataset
        try:
            pubmedqa_df = self._load_file_to_dataframe(self.pubmedqa_path)
            pubmedqa_processed = self._preprocess_pubmedqa(pubmedqa_df)
            datasets["pubmedqa"] = pubmedqa_processed
        except Exception as e:
            logger.error(f"Failed to load and preprocess PubMedQA dataset: {e}")
            raise

        logger.info("Both datasets loaded and preprocessed successfully.")
        return datasets
