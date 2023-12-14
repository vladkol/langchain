"""Vector and document table in Google Cloud BigQuery."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, Literal, List, Optional, Tuple, Type

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)

_ID_COL_NAME = "id"  # document id
_TEXT_EMBEDDING_COL_NAME = "text_embedding"  # embeddings vectors, do not rename
_METADATA_COL_NAME = "metadata"  # document metadata
_CONTENT_COL_NAME = "content"  # text content, do not rename
_TS_COL_NAME = "write_ts"  # document timestamp
_DEFAULT_K = 4  # default number of documents returned from similarity search


class BigQueryVectorStore(VectorStore):
    """Google Cloud BigQuery vector store.

    To use, you need the following packages installed:
        google-cloud-bigquery

    """

    BQ_PY_VECTOR_VERSION = "0_0_1"
    DEFAULT_TABLE_NAME = f"vectors_{BQ_PY_VECTOR_VERSION}"
    DEFAULT_DATASET_NAME = "langchain_vectorstore"

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "US",
        dataset_name: str = DEFAULT_DATASET_NAME,
        table_name: str = DEFAULT_TABLE_NAME,
        embedding: Optional[Embeddings] = None,
        credentials: Optional[Any] = None,
        distance_type: Literal["COSINE", "EUCLIDEAN"] = "COSINE"
    ):
        """Constructor for BigQueryVectorStore

        Args:
            project_id (str, optional): GCP project.
            location (str, optional): BigQuery region. Defaults to `US`
                                                       (multi-region).
            dataset_name (str, optional): BigQuery dataset to store documents
                                and embeddings.
                                Defaults to `langchain_vectorstore`.
                                If doesn't exists, it will be created.
            table_name (str, optional): BigQuery table name.
                                        Defaults to vectors_0_0_1.
            embedding (Embeddings, optional): Embedding model to use.
                                    Defaults to None.
                                    If None, VertexAIEmbeddings will be used.
            credentials (Credentials, optional):
                            Custom Google Cloud credentials to use.
                            Defaults to None.
            distance_type: vector distance function to use,
                           "COSINE" or "EUCLIDEAN". Default is "COSINE".
        """
        try:
            from google.cloud import bigquery

            self.bq_client = bigquery.Client(
                project=project_id, location=location, credentials=credentials
            )
        except ModuleNotFoundError:
            raise ImportError(
                "Please, install or upgrade the google-cloud-bigquery library: "
                "pip install google-cloud-bigquery"
            )
        self.distance_type = distance_type
        self._embedding_function = embedding
        if not self._embedding_function:
            # Figure out region for Vertex AI from BigQuery location.
            # Vertex AI region cannot be a multi-region,
            # but better be within the same multi-region as BigQuery.
            # NOTE: Not all regions are supported by Vertex AI Text Embeddings
            loc_low = location.lower()
            if loc_low == "us":
                ai_location = "us-central1"
            elif loc_low == "eu":
                ai_location = "europe-west1"
            else:
                ai_location = location
            logger.warning(
                "Argument `embedding` is not specified. "
                "Trying to use VertexAIEmbeddings with project `%s` "
                "and location `%s`.",
                project_id,
                ai_location,
            )
            from langchain.embeddings import VertexAIEmbeddings

            self._embedding_function = VertexAIEmbeddings(
                project=project_id, location=ai_location, credentials=credentials
            )
        self.vectors_table = self._initialize_table(dataset_name, table_name)
        self.full_table_id = (
            f"{self.vectors_table.project}."
            f"{self.vectors_table.dataset_id}."
            f"{self.vectors_table.table_id}"
        )

    def _initialize_table(self, dataset_name: str, table_name: str) -> Any:
        """Initializes BigQuery dataset and table."""
        from google.api_core.exceptions import NotFound
        from google.cloud import bigquery

        dataset = self.bq_client.create_dataset(dataset_name, exists_ok=True)
        # If dataset already exists in a different region, we cannot proceed.
        if not dataset.location.lower().startswith( # type: ignore
            self.bq_client.location.lower() # type: ignore
        ):  # type: ignore
            raise ValueError(
                "Existing dataset "
                f"`{self.bq_client.project}.{dataset_name}` "
                "is in a location "
                f"different than `{self.bq_client.location}`."
            )
        table_ref = bigquery.TableReference(dataset.reference, table_name)
        try:
            table = self.bq_client.get_table(table_ref)
        except NotFound:
            table = self.bq_client.create_table(table_ref)
            table.schema = [
                bigquery.SchemaField(
                    name=_ID_COL_NAME, field_type="STRING", mode="REQUIRED"
                ),
                bigquery.SchemaField(
                    name=_CONTENT_COL_NAME, field_type="STRING", mode="NULLABLE"
                ),
                bigquery.SchemaField(
                    name=_METADATA_COL_NAME, field_type="JSON", mode="NULLABLE"
                ),
                bigquery.SchemaField(
                    name=_TEXT_EMBEDDING_COL_NAME, field_type="FLOAT64", mode="REPEATED"
                ),
                bigquery.SchemaField(
                    name=_TS_COL_NAME, field_type="TIMESTAMP", mode="NULLABLE"
                ),
            ]
            self.bq_client.update_table(table, fields=["schema"])
        return table

    def _persist_and_generate(self, data: Dict[str, Any]) -> None:
        """Saves documents to BigQuery.
        If using BigQuery ML updates saved documents with their embeddings."""
        from google.cloud import bigquery

        data_len = len(data[list(data.keys())[0]])
        if data_len == 0:
            return
        time_stamp = datetime.now(timezone.utc)
        ts_str = time_stamp.strftime("%Y-%m-%d %H:%M:%S.%f")  # type: ignore

        data[_TS_COL_NAME] = [ts_str for _ in data[_ID_COL_NAME]]
        list_of_dicts = [dict(zip(data, t)) for t in zip(*data.values())]

        job_config = bigquery.LoadJobConfig()
        job_config.schema = self.vectors_table.schema
        job_config.schema_update_options = (
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        )
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        job = self.bq_client.load_table_from_json(
            list_of_dicts, self.vectors_table, job_config=job_config
        )
        job.result()
        logger.info(
            "%d document(s) have been inserted to the table `%s`.",
            job.output_rows,
            self.full_table_id,
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    @property
    def vector_table_name(self) -> str:
        return self.vector_full_table_id.split(".")[-1]

    @property
    def vector_full_table_id(self) -> str:
        return self.full_table_id

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = [uuid.uuid4().hex for _ in texts]
        if not metadatas:
            metadatas = []
        len_diff = len(ids) - len(metadatas)
        add_meta = [None for _ in range(0, len_diff)]
        metadatas = [m if m is not None else {} for m in metadatas + add_meta]  # type: ignore
        values_dict = {
            _ID_COL_NAME: ids,
            _METADATA_COL_NAME: metadatas,
            _CONTENT_COL_NAME: texts,
        }
        if self._embedding_function:
            embs = self._embedding_function.embed_documents(texts)
            values_dict[_TEXT_EMBEDDING_COL_NAME] = embs
        self._persist_and_generate(values_dict)
        return ids

    def get_documents(
        self, ids: Optional[List[str]] = None, metadata_filter: Optional[str] = None
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field
                             (e.g. `JSON_VALUE(metadata,'$.source') = "Wikipedia"`)
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids and len(ids) > 0:
            from google.cloud import bigquery

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{_ID_COL_NAME} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"
        if metadata_filter and metadata_filter.strip():
            filter_expr = f"AND ({metadata_filter})"
        else:
            filter_expr = ""

        job = self.bq_client.query(
            f"""
                    SELECT * FROM `{self.full_table_id}` WHERE {id_expr}
                    {filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = row[_METADATA_COL_NAME]
            if metadata:
                metadata = json.loads(metadata)
            else:
                metadata = {}
            metadata["__id"] = row[_ID_COL_NAME]
            doc = Document(page_content=row[_CONTENT_COL_NAME], metadata=metadata)
            docs.append(doc)
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if not ids or len(ids) == 0:
            return True
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ids", "STRING", ids),
            ]
        )
        self.bq_client.query(
            f"""
                    DELETE FROM `{self.full_table_id}` WHERE {_ID_COL_NAME}
                    IN UNNEST(@ids)
                    """,
            job_config=job_config,
        ).result()
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def _search_with_score_and_embeddings_by_vector(
        self, embedding: List[float], k: int = 4, metadata_filter: Optional[str] = None
    ) -> List[Tuple[Document, List[float], float]]:
        from google.cloud import bigquery

        if metadata_filter and metadata_filter.strip():
            filter_expr = f"AND ({metadata_filter})"
        else:
            filter_expr = ""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("v", "FLOAT64", embedding),
            ]
        )
        query = f"""
            SELECT
                {_ID_COL_NAME},
                {_METADATA_COL_NAME},
                {_CONTENT_COL_NAME},
                {_TEXT_EMBEDDING_COL_NAME},
            ML.DISTANCE(@v, {_TEXT_EMBEDDING_COL_NAME},
                            '{self.distance_type}') as distance
            FROM `{self.full_table_id}`
            WHERE {_TEXT_EMBEDDING_COL_NAME} IS NOT NULL
                  AND ARRAY_LENGTH({_TEXT_EMBEDDING_COL_NAME}) != 0
                  {filter_expr}
            ORDER BY ABS(distance) ASC
            LIMIT {k}
                """
        document_tuples: List[Tuple[Document, List[float], float]] = []
        job = self.bq_client.query(query,
                                   job_config=job_config,
                                   api_method=bigquery.enums.QueryApiMethod.QUERY)
        for row in job:
            metadata = row[_METADATA_COL_NAME]
            if metadata:
                metadata = json.loads(metadata)
            else:
                metadata = {}
            metadata["__id"] = row[_ID_COL_NAME]
            doc = Document(page_content=row[_CONTENT_COL_NAME], metadata=metadata)
            document_tuples.append(
                (doc, row[_TEXT_EMBEDDING_COL_NAME], row["distance"])
            )
        return document_tuples

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field (e.g. `metadata.source = "web"`)

        Returns:
            List of Documents most similar to the query vector with distance.
        """
        del kwargs
        document_tuples = self._search_with_score_and_embeddings_by_vector(
            embedding, k, metadata_filter
        )
        return [(doc, distance) for doc, _, distance in document_tuples]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field
                             (e.g. `JSON_VALUE(metadata,'$.source') = "Wikipedia"`)

        Returns:
            List of Documents most similar to the query vector.
        """
        tuples = self.similarity_search_with_score_by_vector(
            embedding, k, metadata_filter, **kwargs
        )
        return [i[0] for i in tuples]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = _DEFAULT_K,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with score.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field
                             (e.g. `JSON_VALUE(metadata,'$.source') = "Wikipedia"`)

        Returns:
            List of Documents most similar to the query vector, with similarity scores.
        """
        emb = self._embedding_function.embed_query(query)  # type: ignore
        return self.similarity_search_with_score_by_vector(
            emb, k, metadata_filter, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field
                             (e.g. `JSON_VALUE(metadata,'$.source') = "Wikipedia"`)

        Returns:
            List of Documents most similar to the query vector.
        """
        tuples = self.similarity_search_with_score(query, k, metadata_filter, **kwargs)
        return [i[0] for i in tuples]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.distance_type == "COSINE":
            return BigQueryVectorStore._cosine_relevance_score_fn
        else:
            raise ValueError("Relevance score is not supported "
                             f"for `{self.distance_type}` distance.")

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field
                             (e.g. `JSON_VALUE(metadata,'$.source') = "Wikipedia"`)
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        query_embedding = self._embedding_function.embed_query(  # type: ignore
            query
        )
        doc_tuples = self._search_with_score_and_embeddings_by_vector(
            query_embedding, fetch_k, metadata_filter
        )
        doc_embeddings = [d[1] for d in doc_tuples]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(query_embedding), doc_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [doc_tuples[i][0] for i in mmr_doc_indexes]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            metadata_filter: Filter on metadata values as a WHERE statement expression
                             on `metadata` JSON field
                             (e.g. `JSON_VALUE(metadata,'$.source') = "Wikipedia"`)
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        doc_tuples = self._search_with_score_and_embeddings_by_vector(
            embedding, fetch_k, metadata_filter
        )
        doc_embeddings = [d[1] for d in doc_tuples]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), doc_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [doc_tuples[i][0] for i in mmr_doc_indexes]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""

        func = partial(
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            metadata_filter=metadata_filter,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await asyncio.get_running_loop().run_in_executor(
            None,
            partial(self.max_marginal_relevance_search_by_vector, **kwargs),
            embedding,
            k,
            fetch_k,
            lambda_mult,
            metadata_filter,
        )

    @classmethod
    def from_texts(
        cls: Type["BigQueryVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BigQueryVectorStore":
        """Return VectorStore initialized from texts and embeddings."""
        vs_obj = BigQueryVectorStore(embedding=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas)
        return vs_obj
