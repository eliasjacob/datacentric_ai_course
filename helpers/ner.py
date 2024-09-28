import random
from typing import Dict, Generator, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import skweak
import spacy
from gliner import GLiNER
from skweak.base import SpanAnnotator
from spacy import displacy
from spacy.tokens import Span, Doc
from transformers import pipeline
from langchain_core.runnables.base import RunnableSequence
from langchain_core.pydantic_v1 import BaseModel
import re
import spacy
import skweak.utils

import re
from typing import List, Dict
import spacy
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import numpy as np

class BaseNERAnnotator(SpanAnnotator, ABC):
    """
    Abstract base class for NER annotators.

    Args:
        annotator_name (str): Name of the annotator.
        words_to_skip (Optional[List[str]]): List of words to skip during annotation.
        label_key_name (str): Key name for the entity label in the output.
        merge_adjacent_entities (bool): Whether to merge adjacent entities with the same label.

    Attributes:
        words_to_skip (set): Set of lowercase words to skip during annotation.
        label_key_name (str): Key name for the entity label in the output.
        merge_adjacent_entities (bool): Flag to merge adjacent entities with the same label.
    """

    def __init__(
        self,
        annotator_name: str,
        words_to_skip: Optional[List[str]] = None,
        label_key_name: str = "entity_group",
        merge_adjacent_entities: bool = True,
    ):
        # Initialize the parent class with the annotator name
        super().__init__(annotator_name)
        
        # Convert words to skip to a set of lowercase words for efficient lookup
        self.words_to_skip: set = set(word.lower() for word in (words_to_skip or []))
        
        # Store the key name for the entity label in the output
        self.label_key_name: str = label_key_name
        
        # Flag to determine whether to merge adjacent entities with the same label
        self.merge_adjacent_entities: bool = merge_adjacent_entities

    @abstractmethod
    def find_spans(
        self, doc: Doc
    ) -> Generator[Tuple[int, int, str], None, None]:
        """
        Abstract method to find entity spans in a document.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Yields:
            Tuple[int, int, str]: Start index, end index, and label of each entity span.
        """
        pass

    def _char_to_token_indices(
        self, doc: Doc, start_char: int, end_char: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert character indices to token indices.

        Args:
            doc (Doc): spaCy Doc object.
            start_char (int): Start character index.
            end_char (int): End character index.

        Returns:
            Tuple[Optional[int], Optional[int]]: Start and end token indices, or None if not found.
        """
        # Find the start token index corresponding to the start character index
        start_token = next(
            (
                token.i
                for token in doc
                if token.idx <= start_char < token.idx + len(token.text)
            ),
            None,
        )
        
        # Find the end token index corresponding to the end character index
        end_token = next(
            (
                token.i + 1
                for token in doc
                if token.idx <= end_char <= token.idx + len(token.text)
            ),
            None,
        )
        
        # Return the start and end token indices
        return start_token, end_token

    def _spans_overlap(
        self, span1: Tuple[int, int, str, float], span2: Tuple[int, int, str, float]
    ) -> bool:
        """
        Check if two spans overlap.

        Args:
            span1 (Tuple[int, int, str, float]): First span (start, end, label, score).
            span2 (Tuple[int, int, str, float]): Second span (start, end, label, score).

        Returns:
            bool: True if spans overlap, False otherwise.
        """
        # Check if the maximum start index is less than the minimum end index
        return max(span1[0], span2[0]) < min(span1[1], span2[1])

    def __call__(self, doc: Doc) -> Doc:
        """
        Apply the annotator to a document.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Returns:
            Doc: Annotated spaCy Doc object.
        """
        # Initialize an empty list of spans for the annotator in the document
        doc.spans[self.name] = []
        
        # Iterate over the spans found by the find_spans method
        for start, end, label in self.find_spans(doc):
            # Check if the span is allowed (not in the words_to_skip list)
            if self._is_allowed_span(doc, start, end):
                # Create a Span object and add it to the document's spans
                span = Span(doc, start, end, label=label)
                doc.spans[self.name].append(span)
        
        # Return the annotated document
        return doc

class TransformerNERAnnotator(BaseNERAnnotator):
    """
    NER annotator using a Transformer-based model.

    Args:
        annotator_name (str): Name of the annotator.
        model_name (str): Name of the pretrained transformer model.
        label_mapping (Optional[Dict[str, str]]): Mapping of original labels to new labels.
        score_threshold (float): Minimum confidence score for entity prediction.
        words_to_skip (Optional[List[str]]): List of words to skip during annotation.
        device (int): Device to run the model on (-1 for CPU, >=0 for GPU).
        label_key_name (str): Key name for the entity label in the output.

    Attributes:
        ner_pipeline: Hugging Face NER pipeline.
        label_mapping (Dict[str, str]): Mapping of original labels to new labels.
        score_threshold (float): Minimum confidence score for entity prediction.
    """

    def __init__(
        self,
        annotator_name: str,
        model_name: str,
        label_mapping: Optional[Dict[str, str]] = None,
        score_threshold: float = 0.5,
        words_to_skip: Optional[List[str]] = None,
        device: int = -1,
        label_key_name: str = "entity_group",
    ):
        # Initialize the parent class with the annotator name, words to skip, and label key name
        super().__init__(annotator_name, words_to_skip, label_key_name)
        
        # Initialize the Hugging Face NER pipeline with the specified model and device
        self.ner_pipeline = pipeline(
            "ner", model=model_name, aggregation_strategy="first", device=device
        )
        
        # Store the label mapping, defaulting to an empty dictionary if not provided
        self.label_mapping: Dict[str, str] = label_mapping or {}
        
        # Store the minimum confidence score for entity prediction
        self.score_threshold: float = score_threshold

    def find_spans(
        self, doc: Doc
    ) -> Generator[Tuple[int, int, str], None, None]:
        """
        Find entity spans in a document using the Transformer-based model.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Yields:
            Tuple[int, int, str]: Start index, end index, and label of each entity span.
        """
        # Extract the text from the spaCy document
        text = doc.text
        
        # Use the NER pipeline to predict entities in the text
        predictions = self.ner_pipeline(text)

        spans = []
        for pred in predictions:
            # Filter predictions based on score threshold and words to skip
            if (
                pred["score"] >= self.score_threshold
                and pred["word"].lower() not in self.words_to_skip
            ):
                # Get the character indices for the start and end of the entity
                start_char, end_char = pred["start"], pred["end"]
                
                # Map the original label to a new label if a mapping is provided
                original_label = pred[self.label_key_name]
                mapped_label = self.label_mapping.get(original_label, original_label)

                # Convert character indices to token indices
                start_token, end_token = self._char_to_token_indices(
                    doc, start_char, end_char
                )
                
                # If valid token indices are found, add the span to the list
                if start_token is not None and end_token is not None:
                    spans.append((start_token, end_token, mapped_label, pred["score"]))

        # Sort spans by score (descending) and start position
        spans.sort(key=lambda x: (-x[3], x[0]))

        # Remove overlapping spans
        final_spans = []
        for span in spans:
            # Check if the current span overlaps with any existing span in final_spans
            if not any(
                self._spans_overlap(span, existing_span)
                for existing_span in final_spans
            ):
                final_spans.append(span)

        # Yield the final spans, excluding the score
        for span in final_spans:
            yield span[:3]  # Yield only start, end, and label

class GLiNERAnnotator(BaseNERAnnotator):
    """
    NER annotator using the GLiNER model.

    Args:
        annotator_name (str): Name of the annotator.
        model_name (str): Name of the pretrained GLiNER model.
        labels (List[str]): List of entity labels to predict.
        score_threshold (float): Minimum confidence score for entity prediction.
        words_to_skip (Optional[List[str]]): List of words to skip during annotation.
        label_key_name (str): Key name for the entity label in the output.
        merge_adjacent_entities (bool): Whether to merge adjacent entities with the same label.

    Attributes:
        model: GLiNER model instance.
        labels (List[str]): List of lowercase entity labels to predict.
        score_threshold (float): Minimum confidence score for entity prediction.
    """

    def __init__(
        self,
        annotator_name: str,
        model_name: str,
        labels: List[str],
        score_threshold: float = 0.5,
        words_to_skip: Optional[List[str]] = None,
        label_key_name: str = "label",
        merge_adjacent_entities: bool = True,
    ):
        # Initialize the parent class with the annotator name, words to skip, label key name, and merge flag
        super().__init__(annotator_name, words_to_skip, label_key_name, merge_adjacent_entities)
        
        # Load the pretrained GLiNER model
        self.model = GLiNER.from_pretrained(model_name)
        
        # Store the list of entity labels to predict, converted to lowercase
        self.labels: List[str] = [label.lower() for label in labels]
        
        # Store the minimum confidence score for entity prediction
        self.score_threshold: float = score_threshold

    def find_spans(
        self, doc: Doc
    ) -> Generator[Tuple[int, int, str], None, None]:
        """
        Find entity spans in a document using the GLiNER model.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Yields:
            Tuple[int, int, str]: Start index, end index, and label of each entity span.
        """
        # Extract the text from the spaCy document
        text = doc.text
        
        # Use the GLiNER model to predict entities in the text
        predictions = self.model.predict_entities(
            text, self.labels, threshold=self.score_threshold
        )
        
        # Optionally merge adjacent entities with the same label
        if self.merge_adjacent_entities:
            predictions = merge_adjacent_entities(predictions, text)

        spans = []
        for pred in predictions:
            # Filter predictions based on score threshold and words to skip
            if (
                pred["score"] >= self.score_threshold
                and pred["text"].lower() not in self.words_to_skip
            ):
                # Get the character indices for the start and end of the entity
                start_char, end_char = pred["start"], pred["end"]
                
                # Get the label of the entity
                label = pred[self.label_key_name]

                # Convert character indices to token indices
                start_token, end_token = self._char_to_token_indices(
                    doc, start_char, end_char
                )
                
                # If valid token indices are found, add the span to the list
                if start_token is not None and end_token is not None:
                    spans.append((start_token, end_token, label, pred["score"]))

        # Sort spans by score (descending) and start position
        spans.sort(key=lambda x: (-x[3], x[0]))

        # Remove overlapping spans
        final_spans = []
        for span in spans:
            # Check if the current span overlaps with any existing span in final_spans
            if not any(
                self._spans_overlap(span, existing_span)
                for existing_span in final_spans
            ):
                final_spans.append(span)

        # Yield the final spans, excluding the score
        for span in final_spans:
            yield span[:3]  # Yield only start, end, and label

class LangChainAnnotator(BaseNERAnnotator):
    """
    NER annotator using a LangChain runnable sequence.

    Args:
        annotator_name (str): Name of the annotator.
        langchain_runnable (RunnableSequence): LangChain runnable sequence for NER.
        pydantic_model (BaseModel): Pydantic model for parsing the LangChain output.
        words_to_skip (Optional[List[str]]): List of words to skip during annotation.
        label_key_name (str): Key name for the entity label in the output.
        merge_adjacent_entities (bool): Whether to merge adjacent entities with the same label.

    Attributes:
        model (RunnableSequence): LangChain runnable sequence for NER.
        pydantic_model (BaseModel): Pydantic model for parsing the LangChain output.
    """

    def __init__(
        self,
        annotator_name: str,
        langchain_runnable: RunnableSequence,
        pydantic_model: BaseModel,
        words_to_skip: Optional[List[str]] = None,
        label_key_name: str = "label",
        merge_adjacent_entities: bool = True,
    ):
        # Initialize the parent class with the annotator name, words to skip, label key name, and merge flag
        super().__init__(annotator_name, words_to_skip, label_key_name, merge_adjacent_entities)
        
        # Store the LangChain runnable sequence for NER
        self.model: RunnableSequence = langchain_runnable
        
        # Store the Pydantic model for parsing the LangChain output
        self.pydantic_model: BaseModel = pydantic_model

    def find_spans(
        self, doc: Doc
    ) -> Generator[Tuple[int, int, str], None, None]:
        """
        Find entity spans in a document using the LangChain runnable sequence.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Yields:
            Tuple[int, int, str]: Start index, end index, and label of each entity span.
        """
        # Extract the text from the spaCy document
        text = doc.text
        
        try:
            # Invoke the LangChain model to get predictions
            predictions = self.model.invoke(text)
            
            # Convert the Pydantic model output to GLiNER format
            predictions = self.convert_pydantic_to_gliner_format(text, predictions)
        except Exception as e:
            # Handle any exceptions that occur during model invocation
            print(f"An error occurred: {e}")
            predictions = []

        # Optionally merge adjacent entities with the same label
        if self.merge_adjacent_entities:
            predictions = merge_adjacent_entities(predictions, text)

        spans = []
        for pred in predictions:
            # Filter out predictions for words to skip
            if pred["text"].lower() not in self.words_to_skip:
                # Get the character indices for the start and end of the entity
                start_char, end_char = pred["start"], pred["end"]
                
                # Get the label of the entity
                label = pred[self.label_key_name]

                # Convert character indices to token indices
                start_token, end_token = self._char_to_token_indices(
                    doc, start_char, end_char
                )
                
                # If valid token indices are found, add the span to the list
                if start_token is not None and end_token is not None:
                    spans.append((start_token, end_token, label, pred["score"]))

        # Sort spans by score (descending) and start position
        spans.sort(key=lambda x: (-x[3], x[0]))

        # Remove overlapping spans
        final_spans = []
        for span in spans:
            # Check if the current span overlaps with any existing span in final_spans
            if not any(
                self._spans_overlap(span, existing_span)
                for existing_span in final_spans
            ):
                final_spans.append(span)

        # Yield the final spans, excluding the score
        for span in final_spans:
            yield span[:3]  # Yield only start, end, and label

    def convert_pydantic_to_gliner_format(
        self, text: str, lista_medicamentos: BaseModel
    ) -> List[Dict[str, Any]]:
        """
        Convert Pydantic model output to GLiNER format.

        Args:
            text (str): Original text.
            lista_medicamentos (BaseModel): Pydantic model instance with medication information.

        Returns:
            List[Dict[str, Any]]: List of entity dictionaries in GLiNER format.
        """
        result = []

        # Iterate over each medication in the Pydantic model
        for medicamento in lista_medicamentos.medicamentos:
            # Find all matches of the medication name in the text
            for match in re.finditer(re.escape(medicamento.nome), text, re.IGNORECASE):
                result.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "label": "medicamento",
                        "score": 1.0,
                    }
                )

            # If the medication has an active ingredient, find all matches in the text
            if medicamento.principio_ativo:
                for match in re.finditer(
                    re.escape(medicamento.principio_ativo), text, re.IGNORECASE
                ):
                    result.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "label": "medicamento",
                            "score": 1.0,
                        }
                    )

        # Sort the results by the start index of the matches
        result.sort(key=lambda x: x["start"])
        return result
    
class SaltAnnotator(SpanAnnotator):
    def __init__(self):
        super().__init__("lf_salt")
        self.salt_list = ['21-acetato', 'aceponato', 'acetato', 'acetilsalicilato', 'adipato', 'alendronato', 'alfaoxofenilpropionato', 'alginato', 'aminossalicilato', 'antimoniato', 'arginato', 'arsenito', 'ascorbato', 'aspartato', 'axetil', 'benzoato', 'besilato', 'betacipionato', 'bicarbonato', 'bissulfato', 'bitartarato', 'borato', 'brometo', 'bromidrato', 'butilbrometo', 'caproato', 'carbonato', 'carboxilato', 'ciclossilicato', 'cipionato', 'citrato', 'clatrato', 'clavulanato', 'clonixinato', 'cloranfenicol', 'cloreto', 'cloridrato', 'colistimetato', 'cromacato', 'cromato', 'cromoglicato', 'decanoato', 'di-hidrato', 'diaspartato', 'diatrizoato', 'dicloreto', 'dicloridrato', 'difosfato', 'diidrato', 'dimaleato', 'dimesilato', 'dinitrato', 'dinitrobenzoato', 'dipropionato', 'ditosilato', 'divalproato', 'dobesilato', 'docusato', 'embonato', 'enantato', 'esilato', 'estearato', 'estolato', 'etabonato', 'etanolato', 'etexilato', 'etilsuccinato', 'fempropionato', 'fendizoato', 'fenilpropionato', 'ferededato', 'ferrocianeto', 'fluoreto', 'folinato', 'fosfatidilcolina', 'fosfato', 'fosfito', 'fumarato', 'furoato', 'fusidato', 'gadobenato', 'gadopentetato', 'glicerofosfato', 'glicinato', 'glicirrizato', 'gliconato', 'gluceptato', 'gluconato', 'glutamato', 'hemi-hidrato', 'hemifumarato', 'hemisulfato', 'hemitartarato', 'hexafluoreto', 'hialuronato', 'hiclato', 'hidrobrometo', 'hidrocloreto', 'hidrogenotartarato', 'hidroxibenzoato', 'hidroxinaftoato', 'hipofosfito', 'ibandronato', 'iodeto', 'isetionato', 'isocaproato', 'lactato', 'lactobionato', 'laurato', 'laurilsulfato', 'levolisinato', 'levomalato', 'levulinato', 'lisetil', 'lisina', 'lisinato', 'malato', 'maleato', 'mepesuccinato', 'mesilato', 'metilbrometo', 'metilsulfato', 'metotrexato', 'micofenolato', 'molibdato', 'mono-hidrato', 'monofosfato', 'mononitrato', 'mucato', 'naftoato', 'nicotinato', 'nitrato', 'nitrito', 'nitroprusseto', 'oleato', 'orotato', 'oxalato', 'oxoglurato', 'palmitato', 'pamoato', 'pantotenato', 'pantotênico', 'permanganato', 'piconato', 'picossulfato', 'pidolato', 'pivalato', 'poliestirenossulfonato', 'polissulfato', 'propilenoglicolato', 'propionato', 'racealfa-hidroxigamametiltiobutanoato', 'racealfaoxobetametilbutanoato', 'racealfaoxobetametilpentanoato', 'racealfaoxogamametilpentanoato', 'resinato', 'sacarato', 'salicilato', 'selenato', 'selenito', 'silicato', 'subacetato', 'subgalato', 'succinato', 'sulfato', 'sulfeto', 'sulfito', 'sódico', 'tanato', 'tartarato', 'teoclato', 'tetra-hidrato', 'tiocianato', 'tosilato', 'triclofenato', 'trifenatato', 'undecanoato', 'undecilato', 'undecilenato', 'valerato', 'valproato', 'xinafoato', 'zirconato', 'zíncico']

        self.salt_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.salt_list)) + r')\b', re.IGNORECASE)

    def find_spans(self, doc):
        for i in range(len(doc) - 2):
            if (self.salt_pattern.match(doc[i].text) and
                doc[i + 1].lower_ == 'de' and
                doc[i + 2].is_alpha):
                yield i, i+3, "MEDICAMENTO"

def render_entity_data_from_pipeline(
    text: str,
    pipeline_results: List[Dict[str, Any]],
    colors: Optional[Dict[str, str]] = None,
    label_key_name: str = "entity_group",
) -> None:
    """
    Render entity data from pipeline results using spaCy's displacy.

    Args:
        text (str): Original text.
        pipeline_results (List[Dict[str, Any]]): List of entity predictions from the pipeline.
        colors (Optional[Dict[str, str]]): Color mapping for entity labels.
        label_key_name (str): Key name for the entity label in the pipeline results.

    Returns:
        None
    """
    # Extract entity spans (start, end, label) from pipeline results
    entity_spans = [
        (result["start"], result["end"], result[label_key_name])
        for result in pipeline_results
    ]

    # If no colors are provided, generate random colors for each entity type
    if colors is None:
        # Get unique entity types
        entity_types = list(set([span[2] for span in entity_spans]))
        # Generate random colors for each entity type
        random_colors = get_random_colors(len(entity_types))
        # Map each entity type to a color
        colors = {
            entity_type: random_colors[i] for i, entity_type in enumerate(entity_types)
        }

    # Set displacy options with entity types and their corresponding colors
    displacy_options = {"ents": list(colors.keys()), "colors": colors}

    # Prepare data for displacy rendering
    displacy_data = [
        {
            "text": text,
            "ents": [
                {"start": span[0], "end": span[1], "label": span[2]}
                for span in entity_spans
            ],
        }
    ]

    # Render the entities using spaCy's displacy
    displacy.render(
        displacy_data, style="ent", manual=True, jupyter=True, options=displacy_options
    )

def get_random_colors(num_colors: int) -> List[str]:
    """
    Generate a list of random color hexadecimal codes.

    Args:
        num_colors (int): Number of colors to generate.

    Returns:
        List[str]: List of hexadecimal color codes.
    """
    # Generate a list of random hexadecimal color codes
    return [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(num_colors)
    ]

def merge_adjacent_entities(entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Merge adjacent entities with the same label.

    Args:
        entities (List[Dict[str, Any]]): List of entity dictionaries.
        text (str): Original text.

    Returns:
        List[Dict[str, Any]]: List of merged entity dictionaries.
    """
    # Return an empty list if there are no entities
    if not entities:
        return []

    # Initialize the list of merged entities and set the current entity to the first one
    merged_entities = []
    current_entity = entities[0]

    # Iterate over the remaining entities
    for next_entity in entities[1:]:
        # Check if the next entity is adjacent and has the same label as the current entity
        if next_entity["label"] == current_entity["label"] and (
            next_entity["start"] == current_entity["end"] + 1
            or next_entity["start"] == current_entity["end"]
        ):
            # Merge the current entity with the next entity
            current_entity["text"] = text[
                current_entity["start"] : next_entity["end"]
            ].strip()
            current_entity["end"] = next_entity["end"]
        else:
            # Add the current entity to the list of merged entities and update the current entity
            merged_entities.append(current_entity)
            current_entity = next_entity

    # Add the last current entity to the list of merged entities
    merged_entities.append(current_entity)

    return merged_entities

def extract_entities_in_gliner_format(spacy_doc: spacy.tokens.Doc, annotator_name: str, entities_to_remove: List[str]) -> List[Dict[str, str]]:
    """
    Extract entities from a spaCy document in the format expected by the GLiNER model.

    Args:
        spacy_doc (spacy.tokens.Doc): The spaCy document containing the text and annotations.
        annotator_name (str): The name of the annotator whose annotations are to be used.
        entities_to_remove (List[str]): A list of entity labels to be removed from the text.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the cleaned entity text, its label, and its start and end positions.
    """
    # Initialize an empty list to store the cleaned entities
    cleaned_entities = []
    
    # Get the full text from the spaCy document
    full_text = spacy_doc.text

    # Create a regex pattern to match and remove specified entities
    regex_pattern = '|'.join(map(re.escape, entities_to_remove))
    regex_pattern = r'\b(?:' + regex_pattern + r')\b'

    # Iterate over the annotated spans in the spaCy document
    for span in spacy_doc.spans[annotator_name]:
        # Extract the text corresponding to the current span
        span_text = full_text[span.start_char:span.end_char]
        
        # Remove specified entities from the span text
        cleaned_text = re.sub(regex_pattern, '', span_text, flags=re.IGNORECASE).strip()

        # If the cleaned text is not empty, add it to the list of cleaned entities
        if cleaned_text:
            # Find all matches of the cleaned text within the original span text
            for match in re.finditer(re.escape(cleaned_text), span_text, flags=re.IGNORECASE):
                cleaned_entities.append({
                    "text": cleaned_text,
                    "label": span.label_,
                    "start": span.start_char + match.start(),
                    "end": span.start_char + match.end(),
                })

    return cleaned_entities

def convert_to_IOB(entity_spans: List[Tuple[int, int, str, str]], input_text: str, tokenizer: PreTrainedTokenizer) -> List[Tuple[str, str]]:
    """
    Converts entity spans to IOB format, with a fix for repeating sequences.

    Args:
        entity_spans (List[Tuple[int, int, str, str]]): A list of entity spans. Each span is a tuple of (start, end, text, label).
        input_text (str): The input text.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input text.

    Returns:
        List[Tuple[str, str]]: The list of tokens and their IOB tags.
    """

    tokenized_text = tokenizer(input_text, return_tensors="pt", is_split_into_words=False)
    tokens_int = tokenized_text["input_ids"].squeeze().tolist()
    # # Convert token IDs back to tokens
    # tokens_str = tokenizer.convert_ids_to_tokent(tokens_int)
    # Get word IDs for each token to handle subwords
    word_ids = tokenized_text.word_ids(batch_index=0)  # Assuming a single input for simplicity
    
    # Reconstruct words from token IDs, handling subwords
    tokens = reconstruct_sentence_from_token_ids(
        input_token_ids=tokens_int,
        associated_word_ids=word_ids,
        tokenizer=tokenizer
    )
    
    # Track the positions of the tokens to handle repeating sequences
    token_positions = []
    last_end_position = 0  # Tracks the end position of the last token to disambiguate repeating tokens
    for token in tokens:
        start_position = input_text.find(token, last_end_position)
        end_position = start_position + len(token)
        token_positions.append((start_position, end_position, token))
        last_end_position = end_position  # Update last_end_position to the end of the current token
    
    # Initialize the IOB-tagged output list with 'O' for each token
    iob_tags = ['O'] * len(tokens)
    
    # Process each entity span to assign IOB tags
    for span_start, span_end, matched_text, label in entity_spans:
        start_tagged = False
        for i, (start_position, end_position, token) in enumerate(token_positions):
            if start_position >= span_start and end_position <= span_end:
                if not start_tagged:
                    iob_tags[i] = f'B-{label}'
                    start_tagged = True
                else:
                    iob_tags[i] = f'I-{label}'
    
    # Combine tokens with their IOB tags
    iob_result = [(tokens[i], iob_tags[i]) for i in range(len(tokens))]
    
    return iob_result

def reconstruct_sentence_from_token_ids(input_token_ids: List[int], associated_word_ids: List[int], tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Reconstructs a list of words from token IDs and associated word IDs, handling subwords appropriately.
    
    This function decodes token IDs to their corresponding tokens using a tokenizer. It then iterates through these tokens,
    aggregating subword tokens (prefixed with "##" in BERT-like tokenizers) into their full word forms. Special tokens
    (e.g., [CLS], [SEP] in BERT-like models) are ignored based on their associated word IDs being None.
    
    Args:
        input_token_ids (List[int]): A list of token IDs representing the encoded sentence.
        associated_word_ids (List[int]): A list of word IDs associated with each token. Subword tokens have the same word ID as their preceding tokens.
        tokenizer (PreTrainedTokenizer): The tokenizer used to decode token IDs back to tokens.
    
    Returns:
        List[str]: A list of reconstructed words from the token IDs.
    """
    
    # Decode the list of input token IDs back to their corresponding tokens
    tokens = tokenizer.convert_ids_to_tokens(input_token_ids)
    
    # Initialize an empty list to hold the reconstructed words
    reconstructed_words = []
    # Initialize an empty list to accumulate characters or subwords for the current word
    current_word_fragments = []

    # Iterate through each token and its associated word ID
    for token, word_id in zip(tokens, associated_word_ids):
        if word_id is None:
            # Skip special tokens which do not correspond to any word in the original sentence
            continue
        
        if token.startswith("##"):
            # If the token is a subword (part of a word), remove the "##" prefix and append it to the current word fragments
            current_word_fragments.append(token[2:])
        else:
            # If there's an ongoing word being built (from previous subwords), join its fragments and add to the reconstructed words list
            if current_word_fragments:
                reconstructed_words.append("".join(current_word_fragments))
                current_word_fragments = []  # Reset for the next word
            # Start accumulating fragments for the next word with the current token
            current_word_fragments.append(token)

    # After the loop, check if there's an unfinished word and add it to the reconstructed words list
    if current_word_fragments:
        reconstructed_words.append("".join(current_word_fragments))

    return reconstructed_words

def convert_from_iob_to_gliner_format(iob_tuples: List[Tuple[str, str]], original_text: str) -> List[Dict[str, str]]:
    """
    Convert IOB-tagged text to GLiNER format.

    Args:
        iob_tuples (List[Tuple[str, str]]): A list of tuples where each tuple contains a word and its IOB tag.
        original_text (str): The original text from which the entities are extracted.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the text, label, start, and end positions of an entity.
    """
    # Initialize the result list and variables to track the current entity
    entities = []
    current_entity_words = []
    current_entity_label = None
    entity_start_index = None

    # Iterate over the IOB tuples
    for i, (word, tag) in enumerate(iob_tuples):
        if tag.startswith('B-'):
            # If there is an ongoing entity, finalize it
            if current_entity_words:
                entity_text = ' '.join(current_entity_words)
                entity_end_index = original_text.index(entity_text, entity_start_index) + len(entity_text)
                entities.append({
                    'text': entity_text,
                    'label': current_entity_label,
                    'start': entity_start_index,
                    'end': entity_end_index
                })
            # Start a new entity
            current_entity_words = [word]
            current_entity_label = tag[2:]  # Extract label after 'B-'
            entity_start_index = original_text.index(word, entity_start_index if entity_start_index is not None else 0)
        elif tag.startswith('I-') and current_entity_words and tag[2:] == current_entity_label:
            # Continue the current entity
            current_entity_words.append(word)
        elif tag == 'O':
            # Finalize the current entity if it exists
            if current_entity_words:
                entity_text = ' '.join(current_entity_words)
                entity_end_index = original_text.index(entity_text, entity_start_index) + len(entity_text)
                entities.append({
                    'text': entity_text,
                    'label': current_entity_label,
                    'start': entity_start_index,
                    'end': entity_end_index
                })
                current_entity_words = []
                current_entity_label = None
                entity_start_index = None

    # Handle case where the last word is part of an entity
    if current_entity_words:
        entity_text = ' '.join(current_entity_words)
        entity_end_index = original_text.index(entity_text, entity_start_index) + len(entity_text)
        entities.append({
            'text': entity_text,
            'label': current_entity_label,
            'start': entity_start_index,
            'end': entity_end_index
        })

    return merge_overlapping_named_entities(remove_duplicate_dicts(entities))

    # Example usage
    # iob_tuples = [('Some', 'O'), ('text', 'O'), ('with', 'O'), ('medicamento', 'B-MEDICAMENTO'), ('and', 'O'), ('other', 'O'), ('entities', 'O')]
    # original_text = "Some text with medicamento and other entities."
    # print(convert_from_iob_to_gliner_format(iob_tuples, original_text))

def remove_duplicate_dicts(dict_list: List[Dict]) -> List[Dict]:
    """
    Remove duplicate dictionaries from a list of dictionaries.

    Args:
        dict_list (List[Dict]): A list of dictionaries.

    Returns:
        List[Dict]: A list of dictionaries with duplicates removed.
    """
    # Convert each dictionary to a tuple of its items and use a set to remove duplicates
    seen = set()
    unique_dicts = []
    for d in dict_list:
        # Convert dictionary to a frozenset of its items
        items = frozenset(d.items())
        if items not in seen:
            seen.add(items)
            unique_dicts.append(d)
    return unique_dicts

def merge_overlapping_named_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge overlapping named entities in a list.

    Args:
        entities (List[Dict[str, str]]): A list of dictionaries, each containing the text, label, start, and end positions of an entity.

    Returns:
        List[Dict[str, str]]: A list of merged entities with no overlaps.
    """
    # Sort entities by their start index
    sorted_entities = sorted(entities, key=lambda entity: entity['start'])
    merged_entities = []

    for current_entity in sorted_entities:
        # If merged_entities is empty or current entity doesn't overlap with the last merged entity
        if not merged_entities or current_entity['start'] > merged_entities[-1]['end']:
            merged_entities.append(current_entity)
        else:
            # Overlapping found, update the last merged entity
            last_merged_entity = merged_entities[-1]
            # Update end index if the current entity extends further
            last_merged_entity['end'] = max(last_merged_entity['end'], current_entity['end'])
            # Update text if the current entity is longer
            if len(current_entity['text']) > len(last_merged_entity['text']):
                last_merged_entity['text'] = current_entity['text']

    return merged_entities


def remap_confidence_vectors(
    confidence_vectors: np.ndarray, 
    label_remapping: Dict[int, int]
) -> np.ndarray:
    """
    Remaps the columns of a confidence vector according to a specified remapping.

    This function is useful when you need to aggregate or reassign the probabilities of certain classes
    to new class labels, based on a remapping dictionary. For example, if you have merged several classes
    into a single new class, you can use this function to sum the probabilities of the old classes to
    get the confidence of the new class.

    Args:
        confidence_vectors: A 2D numpy array where each row represents a sample and each column represents
                             the confidence of that sample belonging to a certain class.
        label_remapping: A dictionary where keys are the original class labels and values are the new class
                         labels. The function aggregates probabilities of old labels into the new labels
                         according to this mapping.

    Returns:
        A 2D numpy array of the same height as `confidence_vectors` but potentially different width,
        containing the remapped confidence vectors.

    Raises:
        ValueError: If `confidence_vectors` is not a 2D numpy array.
    """
    # Validate input dimensions
    if confidence_vectors.ndim != 2:
        raise ValueError("confidence_vectors must be a 2D numpy array.")

    # Determine the number of new labels after remapping
    n_new_labels = len(set(label_remapping.values()))
    # Initialize a new confidence matrix with zeros
    new_confidence_vectors = np.zeros((confidence_vectors.shape[0], n_new_labels), dtype=np.float32)

    # Aggregate probabilities for each old label into the new labels
    for old_label, new_label in label_remapping.items():
        new_confidence_vectors[:, new_label] += confidence_vectors[:, old_label]

    return new_confidence_vectors


def correct_label_issues_for_sentence(
    sentence_index: int, 
    all_issues: List[Tuple[int, int]], 
    sentence_tokens_with_iob_labels: List[Tuple[str, str]], 
    predicted_probabilities: List[List[float]], 
    id_to_label_map: Dict[int, str]
) -> List[Tuple[str, str]]:
    """
    Corrects the IOB labels for tokens in a sentence based on identified issues and prediction probabilities.

    Args:
        sentence_index: The index of the sentence being processed.
        all_issues: A list of tuples, where each tuple contains the sentence index and token index of an issue.
        sentence_tokens_with_iob_labels: A list of tuples, where each tuple contains a token and its corresponding IOB label.
        predicted_probabilities: A list of lists containing the prediction probabilities for each token in the sentence.
        id_to_label_map: A dictionary mapping label IDs to their corresponding IOB label strings.

    Returns:
        A list of tuples, where each tuple contains a token and its potentially corrected IOB label.
    """
    # Filter issues specific to the current sentence
    sentence_specific_issues = [issue for issue in all_issues if issue[0] == sentence_index]

    # If there are no issues in the sentence, return the original labels
    if not sentence_specific_issues:
        return sentence_tokens_with_iob_labels
    
    # Extract token indices for the issues in the current sentence
    issue_token_indices = [issue[1] for issue in sentence_specific_issues]

    corrected_labels = []

    # Iterate over each token and its label in the sentence
    for token_index, (token, original_label) in enumerate(sentence_tokens_with_iob_labels):
        # If the current token has an identified issue
        if token_index in issue_token_indices:
            # Get the prediction probabilities for the current token
            token_probs = predicted_probabilities[sentence_index][token_index]
            # Determine the predicted label based on the highest probability
            predicted_label = id_to_label_map[np.argmax(token_probs)]
            # Append the token and its corrected label to the output list
            corrected_labels.append((token, predicted_label))
        else:
            # If no issue, keep the original label
            corrected_labels.append((token, original_label))

    return corrected_labels

def add_iob_tags(tokens_with_labels):
    """
    Add IOB tags (B- and I-) to a list of tuples with tokens and labels.

    Args:
        tokens_with_labels (List[Tuple[str, str]]): List of tuples with tokens and labels without IOB tags.

    Returns:
        List[Tuple[str, str]]: List of tuples with tokens and labels with IOB tags.
    """
    
    tokens_with_labels = [(token, remove_iob_tag(label)) for token, label in tokens_with_labels]

    iob_tokens_with_labels = []
    prev_label = "O"

    for token, label in tokens_with_labels:
        if label == "O":
            iob_tokens_with_labels.append((token, label))
            prev_label = "O"
        else:
            if prev_label != label:
                iob_tokens_with_labels.append((token, "B-" + label))
            else:
                iob_tokens_with_labels.append((token, "I-" + label))
            prev_label = label

    return iob_tokens_with_labels

def remove_iob_tag(label):
    """
    Removes the 'B-' and 'I-' prefixes from an IOB label.

    Args:
        label: The IOB label to process.

    Returns:
        The label without the 'B-' or 'I-' prefix.
    """
    return label.replace('B-', '').replace('I-', '')

