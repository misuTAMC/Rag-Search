# RAG Search — Movie Search & RAG CLI

Một hệ thống tìm kiếm phim chạy trên terminal, kết hợp **keyword search (BM25)**, **semantic search**, **hybrid retrieval**, **LLM re-ranking**, **RAG** và **tìm kiếm bằng ảnh**.

Repository này là một project thực hành các tầng của một hệ thống RAG: từ lập chỉ mục dữ liệu, truy xuất tài liệu, hợp nhất/xếp hạng lại kết quả, đến sinh câu trả lời dựa trên ngữ cảnh tìm được.

## Tính năng

| Nhóm | Chức năng |
| --- | --- |
| Keyword retrieval | Inverted index, TF, IDF, TF-IDF và BM25 tự cài đặt |
| Semantic retrieval | Embedding bằng `all-MiniLM-L6-v2`, cosine similarity và semantic chunking |
| Hybrid retrieval | Weighted score fusion và Reciprocal Rank Fusion (RRF) |
| Query enhancement | Sửa chính tả, rewrite query và mở rộng từ khóa bằng LLM |
| Re-ranking | LLM chấm từng kết quả, LLM batch ranking, hoặc Cross-Encoder |
| RAG | Trả lời câu hỏi, tóm tắt, trả lời có trích dẫn và chat theo ngữ cảnh phim |
| Multimodal | CLIP (`clip-ViT-B-32`) tìm phim bằng ảnh và kết hợp ảnh + text query |
| Ensemble | Gộp kết quả từ BM25, semantic và image retrieval theo consensus |
| Evaluation | Đo Precision@K, Recall@K và F1@K với golden dataset |

## Kiến trúc

```text
Người dùng nhập query / ảnh
        │
        ├── Query enhancement (tùy chọn: spell, rewrite, expand)
        │
        ├── BM25 keyword search
        ├── Chunked semantic search
        └── Image search bằng CLIP (tùy chọn)
                 │
        Hybrid fusion (weighted / RRF) hoặc ensemble consensus
                 │
        Re-rank (LLM / batch LLM / Cross-Encoder, tùy chọn)
                 │
        Kết quả phim hoặc RAG answer / summary / citations / Q&A
```

## Yêu cầu

- Python **3.12+** (project được phát triển với Python 3.12).
- Internet ở lần chạy đầu để tải các model Sentence Transformers/Cross-Encoder và để gọi LLM (nếu dùng tính năng LLM).
- Dataset phim tại `data/movies.json`; xem phần [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu).
- Khuyến nghị dùng [uv](https://docs.astral.sh/uv/) để quản lý môi trường và dependency.

Các thư viện chính đã được khai báo trong `pyproject.toml`:

- `sentence-transformers`, `numpy`, `nltk`: embedding, CLIP và xử lý text.
- `google-genai`, `openai`, `python-dotenv`: Gemini/OpenRouter và biến môi trường.
- `pillow`: đọc ảnh cho multimodal search.
- `rich`: giao diện terminal.

## Cài đặt

### Cách 1 — Dùng uv (khuyến nghị)

```bash
git clone https://github.com/misuTAMC/Rag-Search.git
cd Rag-Search

# Cài uv nếu máy chưa có (macOS)
brew install uv

# Tạo môi trường .venv và cài toàn bộ dependency theo uv.lock
uv sync
```

Nếu không dùng Homebrew, cài `uv` theo hướng dẫn chính thức, sau đó chạy lại `uv sync`.

Mọi ví dụ phía dưới dùng `uv run python ...`, vì lệnh này tự dùng đúng môi trường của project.

### Cách 2 — Dùng venv + pip

```bash
git clone https://github.com/misuTAMC/Rag-Search.git
cd Rag-Search

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install google-genai nltk numpy openai pillow python-dotenv rich sentence-transformers
```

Khi dùng cách này, thay `uv run python` trong các ví dụ bằng `python`.

## Chuẩn bị dữ liệu

> **Lưu ý:** `data/` hiện bị bỏ qua bởi Git, vì vậy dataset và ảnh mẫu không được clone từ GitHub. Bạn cần tự tạo/copy thư mục này trước khi chạy search.

Tối thiểu hãy tạo `data/movies.json`. Hệ thống chấp nhận một JSON array hoặc object có key `movies`. Mỗi phim cần `id`, `title`, `description`.

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Paddington",
      "description": "A young bear from Peru travels to London and loves marmalade."
    },
    {
      "id": 2,
      "title": "The Matrix",
      "description": "A hacker discovers that his reality is a simulated world controlled by machines."
    }
  ]
}
```

Ví dụ tạo dữ liệu tối thiểu để test nhanh:

```bash
mkdir -p data
```

Sau đó tạo file `data/movies.json` với nội dung JSON ở trên. Với dataset đầy đủ, copy file của bạn vào đúng đường dẫn này.

Các file dữ liệu khác:

| File | Bắt buộc | Dùng cho |
| --- | --- | --- |
| `data/movies.json` | Có | Toàn bộ tính năng search/RAG |
| `data/stopwords.txt` | Không | Loại stopword cho keyword search; thiếu file thì hệ thống dùng tập rỗng |
| `data/golden_dataset.json` | Chỉ khi evaluate | Đánh giá Precision/Recall/F1 |
| file ảnh bất kỳ (`.jpg`, `.jpeg`, `.png`) | Chỉ khi image search | Multimodal search và describe-image |

Cache được tạo tự động trong `cache/`: inverted index, BM25 statistics, document embeddings và chunk embeddings. Xóa cache khi thay dataset để các chỉ mục được tạo lại:

```bash
rm -rf cache
```

## Cấu hình LLM (tùy chọn)

Các chức năng rewrite, rerank, RAG, summary, citations, Q&A và describe-image cần ít nhất một API key. Tạo file `.env` tại thư mục gốc project:

```dotenv
# Có thể dùng một hoặc cả hai key
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

Code ưu tiên/fallback giữa Gemini (`gemini-2.5-flash`) và OpenRouter. Không commit file `.env` hoặc API key lên GitHub.

## Chạy nhanh

Đứng tại thư mục gốc repo. Với dataset mới, nên chạy build index một lần:

```bash
uv run python cli/keyword_search_cli.py build
```

Sau đó thử tìm theo từ khóa:

```bash
uv run python cli/keyword_search_cli.py bm25search "bear marmalade"
```

Lần đầu chạy semantic/hybrid/image search có thể mất thời gian do tải model và tạo embedding cache.

## Hướng dẫn các CLI

Mỗi lệnh đều có trợ giúp tích hợp:

```bash
uv run python cli/hybrid_search_cli.py --help
uv run python cli/hybrid_search_cli.py rrf_search --help
```

### 1. Keyword search và BM25

```bash
# Tạo/lưu inverted index
uv run python cli/keyword_search_cli.py build

# Search theo token; kết quả đơn giản
uv run python cli/keyword_search_cli.py search "space adventure"

# Search dùng BM25
uv run python cli/keyword_search_cli.py bm25search "space adventure"

# Kiểm tra các thành phần điểm số
uv run python cli/keyword_search_cli.py tf 1 bear
uv run python cli/keyword_search_cli.py idf bear
uv run python cli/keyword_search_cli.py tfidf 1 bear
uv run python cli/keyword_search_cli.py bm25idf bear
uv run python cli/keyword_search_cli.py bm25tf 1 bear 1.5 0.75
```

Text được lowercase, bỏ punctuation/stopword và stem bằng Porter Stemmer trước khi index/query.

### 2. Semantic search và chunking

```bash
# Kiểm tra model và embedding
uv run python cli/semantic_search_cli.py verify
uv run python cli/semantic_search_cli.py embed_text "family adventure movie"
uv run python cli/semantic_search_cli.py embed_query "cute bear in London"

# Chunk theo số từ hoặc theo câu
uv run python cli/semantic_search_cli.py chunk "First sentence. Second sentence." --chunk_size 5 --overlap 1
uv run python cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence. Third sentence." --max_chunk_size 2 --overlap 1

# Tạo vector cho chunks, sau đó semantic search
uv run python cli/semantic_search_cli.py embed_chunks
uv run python cli/semantic_search_cli.py search_chunked "cute british bear marmalade" --limit 5
```

Semantic chunking chia mô tả theo câu (mặc định tối đa 4 câu/chunk, overlap 1 câu), tìm chunk gần query nhất rồi lấy điểm tốt nhất của mỗi phim.

### 3. Hybrid search

```bash
# Weighted score fusion
uv run python cli/hybrid_search_cli.py weighted_search "family adventure" --alpha 0.5 --limit 5

# Reciprocal Rank Fusion (RRF)
uv run python cli/hybrid_search_cli.py rrf_search "cute british bear marmalade" --limit 5

# Điều chỉnh hằng số làm mượt k của RRF
uv run python cli/hybrid_search_cli.py rrf_search "science fiction hacker" -k 60 --limit 10

# Chuẩn hóa một danh sách điểm
uv run python cli/hybrid_search_cli.py normalize 0.2 0.5 0.9
```

Trong implementation hiện tại của `weighted_search`, `alpha` là trọng số của **BM25** và `1 - alpha` là trọng số của semantic score. RRF hợp nhất theo công thức `1 / (k + rank)`.

#### Hybrid + query enhancement / re-ranking

Các lệnh dưới đây cần API key cho phương án LLM; `cross_encoder` sẽ tải model riêng ở lần đầu.

```bash
# Sửa lỗi chính tả, rewrite hoặc expand query trước retrieval
uv run python cli/hybrid_search_cli.py rrf_search "paddinton bear" --enhance spell --limit 5
uv run python cli/hybrid_search_cli.py rrf_search "bear movie in london" --enhance rewrite --limit 5
uv run python cli/hybrid_search_cli.py rrf_search "funny bear film" --enhance expand --limit 5

# Re-rank kết quả RRF
uv run python cli/hybrid_search_cli.py rrf_search "family bear movie" --rerank-method individual --limit 5
uv run python cli/hybrid_search_cli.py rrf_search "family bear movie" --rerank-method batch --limit 5
uv run python cli/hybrid_search_cli.py rrf_search "family bear movie" --rerank-method cross_encoder --limit 5

# Dùng LLM đánh giá độ liên quan cuối cùng theo thang 0–3
uv run python cli/hybrid_search_cli.py rrf_search "family bear movie" --evaluate --limit 5
```

### 4. RAG, summary, citations và hỏi–đáp

Các command này retrieve bằng RRF trước, rồi gửi các kết quả tìm được làm context cho LLM.

```bash
# Trả lời theo context phim
uv run python cli/augmented_generation_cli.py rag "Which movies are about a bear in London?"

# Tổng hợp nhiều kết quả
uv run python cli/augmented_generation_cli.py summarize "family bear movies" --limit 5

# Câu trả lời có đánh số nguồn [1], [2], ...
uv run python cli/augmented_generation_cli.py citations "Compare bear movies for children" --limit 5

# Hỏi đáp theo dữ liệu phim
uv run python cli/augmented_generation_cli.py question "Who directed Dark Star?" --limit 5
```

LLM được prompt để chỉ dựa trên documents đã retrieve. Nếu context không đủ, câu trả lời nên nêu rõ điều đó.

### 5. Image / multimodal search

```bash
# Kiểm tra vector ảnh từ CLIP
uv run python cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg

# Tìm 5 phim gần với ảnh nhất
uv run python cli/multimodal_search_cli.py image_search data/paddington.jpeg

# Dùng vision model của OpenRouter để viết lại text query dựa trên ảnh + query
uv run python cli/describe_image_cli.py --image data/paddington.jpeg --query "a movie like this"
```

Image search encode title + description của toàn bộ phim vào CLIP text space, encode ảnh query vào cùng space, sau đó xếp hạng theo cosine similarity.

### 6. Ensemble consensus search

Ensemble chạy BM25 và semantic search; nếu truyền `--image`, nó thêm image search. Một phim xuất hiện trong càng nhiều engine và ở vị trí càng cao sẽ có điểm consensus càng tốt.

```bash
# Ensemble text search
uv run python cli/ensemble_search_cli.py ensemble_search "cute british bear marmalade"

# Thêm enhancement/re-rank
uv run python cli/ensemble_search_cli.py ensemble_search "paddinton bear" --enhance spell --rerank-method batch
uv run python cli/ensemble_search_cli.py ensemble_search "family bear movie" --rerank-method cross_encoder --evaluate

# Kết hợp ảnh và text
uv run python cli/ensemble_search_cli.py ensemble_search "bear in London" --image data/paddington.jpeg
```

Ở ensemble implementation hiện tại, `batch` và `cross_encoder` là hai re-rank method có xử lý riêng; `individual` có xuất hiện trong lựa chọn CLI nhưng chưa được áp dụng như một bước re-rank riêng cho ensemble.

### 7. Evaluation

Tạo `data/golden_dataset.json` theo định dạng sau trước khi đánh giá:

```json
{
  "test_cases": [
    {
      "query": "cute british bear marmalade",
      "relevant_docs": ["Paddington"]
    }
  ]
}
```

Sau đó chạy:

```bash
uv run python cli/evaluation_cli.py --limit 5
```

Chương trình dùng RRF, in chỉ số của từng query và trung bình `Precision@K`, `Recall@K`, `F1@K`, cùng tổng thời gian đánh giá.

## Cấu trúc thư mục

```text
.
├── cli/
│   ├── keyword_search_cli.py       # CLI cho index, TF/IDF/BM25
│   ├── semantic_search_cli.py      # CLI embedding, chunking, semantic search
│   ├── hybrid_search_cli.py        # Weighted/RRF hybrid search
│   ├── augmented_generation_cli.py # RAG, summary, citations, Q&A
│   ├── multimodal_search_cli.py    # Tìm kiếm theo ảnh
│   ├── ensemble_search_cli.py      # Consensus từ nhiều retrieval engine
│   ├── evaluation_cli.py            # CLI đánh giá
│   └── lib/
│       ├── keyword_search.py
│       ├── semantic_search.py
│       ├── hybrid_search.py
│       ├── llm.py
│       ├── rag_llm_command.py
│       ├── multimodal_search.py
│       ├── ensemple_consensus_search.py
│       └── prompts/                 # Prompt templates cho LLM
├── data/                            # Dataset cục bộ, hiện không được Git track
├── cache/                           # Index/vector cache tự sinh
├── pyproject.toml                   # Python dependencies
└── uv.lock                          # Dependency lock file
```
## Ví dụ về sử dụng search
### 1.
<img width="1218" height="1085" alt="Ảnh màn hình 2026-08-12 lúc 07 46 52" src="https://github.com/user-attachments/assets/1bdb5a25-a1a1-4d7a-9376-41c5b496c1cc" />
### 2.
<img width="1215" height="436" alt="Ảnh màn hình 2026-08-12 lúc 07 53 17" src="https://github.com/user-attachments/assets/9dc57d5c-53af-4d77-a86b-b7d16965dbc6" />
### 3.
<img width="1220" height="475" alt="Ảnh màn hình 2026-08-12 lúc 07 55 11" src="https://github.com/user-attachments/assets/0be1ab6c-493a-4b41-a6af-173ae93ec7a8" />
### 4. 
- nếu description không có thông tin quá rõ thì do LLM đã bị ép do prompt, chỉ dựa vào description để trả lời,nếu không đủ thì trả về không có thông tin tránh hallucination
<img width="1222" height="525" alt="Ảnh màn hình 2026-08-12 lúc 07 57 56" src="https://github.com/user-attachments/assets/9d67aff0-0d35-44d4-9fd3-592d933313c2" />
