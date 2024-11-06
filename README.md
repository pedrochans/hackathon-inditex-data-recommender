# Recomender System 

Category   ➡️   Data Science

Subcategory   ➡️   Recommender systems

Difficulty   ➡️   Medium

Expected solution time ➡️ 8 hours. It is essential to complete your solution within this timeframe, as it is a critical performance indicator used by the hiring company to evaluate your work. The timer will begin when you click the start button and will stop upon your submission.

---

## 🌐 Background
As e-commerce and digital retail continue to grow,  personalizing the shopping experience has become essential to improving customer satisfaction and increasing sales. One effective way to achieve this is through recommender systems, which help suggest products tailored to individual customer preferences. This system would analyze data such as browsing history, previous purchases, demographic details, and product attributes. Unlike many products, clothing choices are often influenced by both personal style and external trends, which can change quickly. Additionally, factors like size, color, brand loyalty, and even occasion-specific requirements can play a crucial role in a customer’s choice. A well-designed recommendation model can enhance user engagement, improve conversion rates, and build long-term customer loyalty.

### 🗂️ Dataset 

There will be three different datasets available:

- Users:
  - Variables:
    - `user_id`: User identifier.
    - `country`: Country identifier.
    - `R`: User's recency.
    - `F`: User's purchase frequency.
    - `M`: Monetary value that the customer spends on purchases.
  - Data available though api endpoint:
    -  API Endpoints: `https://zara-boost-hackathon.nuwe.io/docs`
    -  Parameters required: `user_id`
    -  Result: A dictionary with client details as values. Each parameter is returned as a list as a user may have more than one entry.

``` 
{
  "user_id":  "1556",
  "values":
    {
      "country":[25],
      "R": [26],
      "F": [1979], 
      "M": [198], 
    }
}
```

- Products: Data of all the available products 
  - Variables:
    - `discount`: Boolean indicating if the product is on sale.
    - `embedding`: Embedding of the product's flat image obtained from computer vision techniques.
    - `partnumber`: Product identifier.
    - `color_id`: Product color identifier.
    - `cod_section`: Section to which the product belongs.
    - `familiy`: Product family to which the product belongs.
  - Data available at: [Download product data](https://cdn.nuwe.io/challenges-ds-datasets/hackathon-inditex-data-24/products.zip)
- Train, Test:
  -  Variables:
      - `session_id`: Session identifier.
      - `date`: Interaction date.
      - `timestamp_local`: Interaction timestamp.
      - `user_id`: User identifier.
      - `country`: Country identifier.
      - `partnumber`: Product identifier with which the interaction occurred.
      - `device_type`: Type of device used.
      - `pagetype`: Type of page where the interaction occurred within the e-commerce site.
      - `add_to_cart`: Boolean indicating if the interaction was adding to the cart. This variable won't be available in the test dataset.

  - Datasets are available via direct link download:
    - [Download train data](https://cdn.nuwe.io/challenges-ds-datasets/hackathon-inditex-data-24/train.zip)
    - [Download test data](https://cdn.nuwe.io/challenges-ds-datasets/hackathon-inditex-data-24/test.zip)
  


### 📊 Data Processing

Implement data processing for a recommender system to clean, filter, and normalize user and item data, removing noise and irrelevant information. Process features by transforming raw inputs—such device type, product embeddings, and purchase history—into structured formats that highlight patterns and relationships. Aggregate and scale these features to feed into machine learning models, optimizing for personalized, accurate recommendations based on individual user behaviors.

### 🤖 Model

Develop a recommender system capable of recommending **5 products** (`partnumber`) to a user. Take into account that the user may be logged in or not, that they may have had some previous interactions within the plaform, and any other aspect you may consider relevant.

## 📂 Repository Structure
The repository structure is provided and must be adhered to strictly:

```

├── data/                      
│   ├── raw/
│   └── processed/              
│
│── predictions/   
│   ├── predictions_1.json 
│   └── predictions_3.json      
│
├── src/                       
│   ├── data/                   
│   │   ├── api_calls.py       
│   │   ├── data_functions.py       
│   │   └── data_questions.py     
│   │
│   └── models/
│       ├── prepare_data.py      
│       ├── train_model.py      
│       └── predict_model.py         
│
├── tests/                      
│   ├── function_test.py       
│   └── conftest.py                    
│
├── README.md  
└── requirements.txt     

```

The `/predictions` folder should contain the tasks outputs for the questions in Task 1 and the predictions for Task 2.
 

## 🎯 Tasks
This challenge will include two tasks: an initial exploratory task with questions and a function creation to assess general query knowledge, followed by a second task focused on creating a recommender system. The primary focus will be on the second task.

- **Task 1:** Answer the following questions and develop two functions about the train, clients and products datasets:
    - **Q1:** Which product (`partnumber`) with `color_id` equal to 3   belongs to the lowest `familiy` code with a `discount`? 
    - **Q2:** In the country where most users have made purchases totaling less than 500 (`M`) , which is the user who has the lowest purchase frequency (`F`), the most recent purchase (highest `R`) and the lowest `user_id`? Follow the given order of variables as the sorting priority.
    - **Q3:** Among the sessions where exactly 18 products were added to the cart in one session, lowest session that has item 1328 in the cart, with this item being the lowest `partnumber` among the items added to the cart.
    - **Q4:** Which device (`device_type`) is most frequently used by users to make purchases (`add_to_cart` = 1) of discounted products (`discount` = 1)?
    - **Q5:**Among users with purchase frequency (`F`) in the top 3 within their purchase country, who has interacted with the most products (`partnumber`) in sessions conducted from a device with identifierr 3 (`device_type` = 3)?
    - **Q6:** For interactions that occurred outside the user's country of residence, how many unique family identifiers are there?
    - **Q7:** Among interactions from the first 7 days of June by users with purchases value under 100, in which session is the sum of embeddings for all products they interacted with the lowest?
- **Task 2:** Develop the following function:
    - **Function:** Given a DataFrame with the format of the TRAIN dataset, return a DataFrame with the following columns: user identifier (`user_id`), session identifier (`session_id`), total session duration in seconds (`total_session_time`), and the percentage of products added to the cart out of the total products interacted with (`cart_addition_ratio`). The result should be sorted in ascending order by user identifier and then by session identifier.
```python
def get_session_metrics(df: pd.DataFrame,user_id: int)-> pd.DataFrame:
	return pd.DataFrame(columns = ["user_id","session_id", "total_session_time", "cart_addition_ratio"])
```

- **Task 3:**
  You are tasked with building a **recommendation system** to suggest **five products** for each session ID. 

  You are provided with data on products, customers, and a training dataset (Train). The test dataset (Test) is balanced across four customer types:

  - New customers who start their activity on the platform without prior interactions in the session:
    - Customers present in the Train dataset.
    - Customers not present in the Train dataset.
  - Returning customers who have already engaged in some interactions within the current session (some initial interactions from the session will be provided):
    - Customers present in the Train dataset.
    - Customers not present in the Train dataset.


### 💫 Guides
Ensure that the project environment is configured to use Python 3.12. The libraries listed in `requirements.txt` must not be altered, although you are permitted to add additional libraries as needed.

## 📤 Submission

Submit a `predictions_1.json` file for the queries in task 1 and a `predictions_2.json` file containing the recommendations made by your model. Ensure the file is formatted correctly, you are provided with the example submission files in the folder `/predictions` .


## 📊 Evaluation
- **Task 1:** The questions of this tasks will be evaluated via JSON file, comparing your answer in `predictions_1.json` against the expected value.
- **Task 2:** The function will be evaluated via PyTest. You are provided with some sample tests in the `/tests` folder, take into account that for the evaluation the tests will be a different subset not provided. 

- **Task 3** 
The system's performance will be evaluated using the **Normalized Discounted Cumulative Gain (NDCG)** metric using your predictions in `predictions_3.json` against the true actions of the user.
- **Ranking Criteria:**
  - **Relevance:** The relevance score will be a binary label based on whether or not the product was ultimately added to the cart.
The test dataset provided excludes the `add_to_cart` variable, as it will be used for recommendation evaluation. Additionally, the given test dataset only includes users who have added at least five products to their cart. Note that in the cases where you are provided with some interactions within the session, those products may or may not have been added to the cart.

The grading system will be the following:

- Task 1: 100 / 1100 points
- Task 2: 100 / 1100 points
- Task 3: 900 / 1100 points

**⚠️ Please note:**  
All submissions might undergo a manual code review process to ensure that the work has been conducted honestly and adheres to the highest standards of academic integrity. Any form of dishonesty or misconduct will be addressed seriously, and may lead to disqualification from the challenge.

Ensure that all data manipulation and model training strictly utilize the libraries mentioned in `requirements.txt`.


## ❓ FAQs

**Q1: How do I run the sample tests?**
A1: In the command line, write the command `python -m pytest tests/function_tests.py` for Task 2. Ensure the `train.csv` file is located in the `/data/raw/` directory.

**Q2: How do I submit my solution?**
A2: Submit your solution via Git. Once your code and predictions are ready,commit your changes to the main branch and push your repository. Your submission will be graded automatically within a few minutes. Make sure to write meaningful commit messages.