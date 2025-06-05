from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import torch
import wandb
from tqdm import tqdm 
import argparse
import os
import logging
import sys
import gc
from pathlib import Path

# Predefined configurations
CONFIGURATIONS = [
    {'batch_size': 4, 'learning_rate': 5e-5},
    {'batch_size': 4, 'learning_rate': 2e-5},
    {'batch_size': 8, 'learning_rate': 5e-5},
    {'batch_size': 8, 'learning_rate': 2e-5},
    {'batch_size': 16, 'learning_rate': 5e-5},
    {'batch_size': 16, 'learning_rate': 2e-5},
]

# Set up logging to both file and console
def setup_logging(output_dir):
    log_file = Path(output_dir) / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Set tokenizer parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set CUDA device order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='FacebookAI/xlm-roberta-base', type=str, help='model id')
parser.add_argument('--cache_dir', default=None, type=str, help='model cache dir')
parser.add_argument('--wandb_proj_name', default="mmsd_all", type=str, help='wandb project name')
parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs')
parser.add_argument('--use_single_gpu', action='store_true', help='use only a single GPU')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use when use_single_gpu is True')
parser.add_argument('--output_dir', default="sweep", type=str, help='output directory')
parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='path to checkpoint to resume from')
parser.add_argument('--config_index', type=int, help='index of configuration to use from CONFIGURATIONS')
parser.add_argument('--train_file', type=str, required=True, help='path to training file')
parser.add_argument('--valid_file', type=str, required=True, help='path to validation file')
parser.add_argument('--test_file', type=str, required=True, help='path to test file')
parser.add_argument('--test_only', action='store_true', help='only run testing without training')
parser.add_argument('--model_path', type=str, help='path to the model to use for testing')
args = parser.parse_args()

model_id = args.model_id
model_name = model_id.split("/")[-1]
cache_dir = args.cache_dir
output_dir = Path(args.output_dir) / model_name
wandb_proj_name = args.wandb_proj_name

# Set up logging
logger = setup_logging(output_dir)

# Set GPU device if using single GPU
if args.use_single_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger.info(f"Using single GPU: {args.gpu_id}")
else:
    logger.info("Using all available GPUs")

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
epoch = args.num_epochs

def cleanup_gpu():
    """Clean up GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc, "f1-macro": f1_macro}

def load_data():
    """Load and preprocess the dataset"""
    logger.info("Loading training data...")
    try:
        # Load train, validation and test sets
        train_df = pd.read_csv(args.train_file, sep=",", encoding="utf-8")
        valid_df = pd.read_csv(args.valid_file, sep=",", encoding="utf-8")
        test_df = pd.read_csv(args.test_file, sep=",", encoding="utf-8")

        # For all sets, combine text and image_description
        train_df['combined_text'] = train_df['text'] + ' ' + train_df['image_description']
        valid_df['combined_text'] = valid_df['text'] + ' ' + valid_df['image_description']
        test_df['combined_text'] = test_df['text'] + ' ' + test_df['image_description']

        # Rename columns to match the structure
        train_df = train_df.rename(columns={"text": "only_text", "combined_text": "text"})
        valid_df = valid_df.rename(columns={"text": "only_text", "combined_text": "text"})
        test_df = test_df.rename(columns={"text": "only_text", "combined_text": "text"})

        # Select and clean data
        train_df = train_df[["text", "label"]].copy()
        valid_df = valid_df[["text", "label"]].copy()
        test_df = test_df[["text", "label"]].copy()

        train_df.dropna(inplace=True)
        valid_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        logger.info(f"Train size: {train_df.shape[0]}, Val size: {valid_df.shape[0]}, Test size: {test_df.shape[0]}")
        
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(valid_df),
        })

        try:
            dataset = dataset.remove_columns(["__index_level_0__"])
        except: 
            logger.info("index column does not exist in the dataset!")

        return dataset, test_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def init_model(config=None):
    try:
        with wandb.init(config=config, project=wandb_proj_name) as run:
            config = wandb.config
            run_name = f"model_id:{model_name}|batch:{config.batch_size}|lr:{config.learning_rate}"
            params = {
                "batch_size": config.batch_size, 
                "lr": config.learning_rate, 
                "run_name" : run_name
            }
            run.name = run_name
            train(params)
    except Exception as e:
        logger.error(f"Error in init_model: {str(e)}")
        raise

def train(params):
    global sweep_no
    sweep_no += 1
    
    try:
        logger.info(f"Starting training with batch size: {params['batch_size']}, learning rate: {params['lr']}")
        
        # Clean up GPU memory before loading model
        cleanup_gpu()
        
        # Load model without specifying device_map to allow Trainer to handle device placement
        logger.info("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, id2label=id2label, label2id=label2id, cache_dir=cache_dir
        )
        
        # Make parameters contiguous for better performance
        for p in model.parameters():
            p.data = p.data.contiguous()
        
        # Configure training arguments
        logger.info("Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir=str(output_dir / f"run_{sweep_no}"),
            optim="adamw_torch",
            logging_steps=1,
            learning_rate=params["lr"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["batch_size"],
            num_train_epochs=epoch,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_steps=10,
            eval_steps=10,
            report_to="wandb",
            dataloader_num_workers=0,
            run_name=params["run_name"],
            load_best_model_at_end=False,
            seed=42,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            ddp_find_unused_parameters=False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            logging_dir=str(output_dir / "logs"),
            logging_first_step=True,
            save_total_limit=2,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logger.info("Training completed. Saving model...")
        trainer.save_model()
        logger.info("Evaluating model...")
        evaluate(trainer.model, params["run_name"])
        
        # Clean up after training
        cleanup_gpu()
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

def evaluate(model, run_name):
    global result_dict
    global test_df
    
    try:
        logger.info("Starting evaluation...")
        
        # Create a test dataset for efficient batch processing
        test_dataset = Dataset.from_pandas(test_df)
        logger.info("Tokenizing test dataset...")
        tokenized_test = test_dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding=True),
            batched=True,
            remove_columns=["text"]
        )
        
        # Use Trainer for efficient evaluation
        logger.info("Creating evaluation trainer...")
        eval_trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics,
        )
        
        # Get predictions in batches
        logger.info("Running predictions...")
        predictions = eval_trainer.predict(tokenized_test).predictions
        predictions = np.argmax(predictions, axis=1)
        references = test_df["label"].to_list()
        
        logger.info("Generating classification report...")
        print(classification_report(references, predictions, labels=[0, 1]))
        
        conf_matrix = str(confusion_matrix(references, predictions).tolist())
        metrics = {
            "f1-macro" : f1_score(references, predictions, average='macro'),
            "f1-micro" : f1_score(references, predictions, average='micro'),
            "f1-weighted" : f1_score(references, predictions, average='weighted'),
            "precision" : precision_score(references, predictions, average='weighted'),
            "recall" : recall_score(references, predictions, average='weighted'),
            "accuracy" : accuracy_score(references, predictions)
        }
        
        result_dict["confusion matrix"].append(conf_matrix)
        result_dict["metrics"].append(metrics)
        result_dict["run name"].append(run_name)
        result_dict["f1-macro"].append(metrics["f1-macro"])
        
        logger.info("Evaluation completed.")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

# Load data and tokenizer
dataset, test_df = load_data()
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "non-satiric", 1: "satiric"}
label2id = {"non-satiric": 0, "satiric": 1}

# Start sweep
result_dict = {"run name":[], "metrics": [], "confusion matrix": [], "f1-macro":[]}
sweep_no = 0

if __name__ == "__main__":
    try:
        logger.info("Starting main execution...")
        
        if args.test_only:
            if not args.model_path:
                raise ValueError("model_path must be specified when using test_only mode")
            
            logger.info("Running in test-only mode...")
            # Load model for testing
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_path, num_labels=2, id2label=id2label, label2id=label2id
            )
            
            # Run evaluation
            evaluate(model, "test_only_run")
            
            # Save results
            result_df = pd.DataFrame.from_dict(result_dict)
            print("\nTest Results:")
            print(result_df)
            result_df.to_csv(output_dir / f"test_results_{model_name}.csv", sep="\t", encoding="utf-8")
            
        else:
            # Original training code
            if args.config_index is not None:
                if args.config_index < 0 or args.config_index >= len(CONFIGURATIONS):
                    raise ValueError(f"config_index must be between 0 and {len(CONFIGURATIONS)-1}")
                config = CONFIGURATIONS[args.config_index]
                logger.info(f"Running configuration {args.config_index}: {config}")
                init_model(config)
            else:
                logger.info("Running all configurations...")
                for config in CONFIGURATIONS:
                    init_model(config)
            
            result_df = pd.DataFrame.from_dict(result_dict)
            print(result_df)
            result_df.to_csv(output_dir / f"sweep_{model_name}.csv", sep="\t", encoding="utf-8", mode='a', header=False)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise 