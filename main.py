import openai
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

openai.api_key = "API_KEY"

# openai.api_key = open("API_KEY", "r").read()

def get_stock_price(ticker): # get stock price for the day for the given ticker = symbol of interest 
    data = yf.Ticker(ticker).history(period="1y").Close # get stock info for the ticker
    return str(data) # return stock price for the day


def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period="1y").Close # get stock info for the last year
    return str(data.rolling(window=window).mean().iloc[-1]) # calculate SMA for the given window

def calculate_EMA(ticker,window):
    data = yf.Ticker(ticker).history(period="1y").Close # get stock price for the last year
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1]) # calculate EMA for the given window

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history( period="1y").Close # get stock price for the last year
    delta = data.diff() # get price difference
    up = delta.clip(lower=0) # get positive price difference
    down = -1*delta.clip(upper=0) # get negative price difference
    ema_up = up.ewm(com=14-1, adjust=False).mean() # calculate EMA for positive price difference
    ema_down = down.ewm(com=14-1, adjust=False).mean() # calculate EMA for negative price difference
    rs = ema_up/ema_down # calculate RS
    return str(100 - (100/(1+rs)).iloc[-1]) # calculate RSI 

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period="1y").Close # get stock price for the last year
    shortEMA = data.ewm(span=12, adjust=False).mean() # calculate short EMA
    longEMA = data.ewm(span=26, adjust=False).mean() # calculate long EMA

    MACD = shortEMA - longEMA # calculate MACD
    signal = MACD.ewm(span=9, adjust=False).mean() # calculate signal
    MACD_histogram = MACD - signal # calculate MACD histogram
    
    return f"{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}" # return MACD, signal and MACD histogram values

def plot_stock_price(ticker):
    stock = yf.Ticker(ticker).history(period="1y") # get stock price for the last year
    plt.figure(figsize=(10, 5)) # set figure size
    plt.plot(stock.index, stock.Close)
    plt.title(f"{ticker} Stock Price Over Last Year ")  # set title

    plt.xlabel("Date") # set x-axis label
    plt.ylabel(" Stock Price ($)") # set y-axis label
    plt.grid(True) # add grid
    plt.savefig("stock_price.png") # save plot as PNG file
    plt.close()   # close plot


functions = [
    {
        "name": "calculate_SMA",
        "description": "Calculate the Simple Moving Average (SMA) for the given stock ticker and window size",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., 'AAPL')"
                },
                "window": {
                    "type": "integer",
                    "description": "The window size for the SMA calculation"
                }
            },
            "required": ["ticker", "window"]
        }
    },
    {
        "name": "calculate_EMA",
        "description": "Calculate the Exponential Moving Average (EMA) for the given stock ticker and window size",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., 'AAPL')"
                },
                "window": {
                    "type": "integer",
                    "description": "The window size for the EMA calculation"
                }
            },
            "required": ["ticker", "window"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for the given ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., 'AAPL')"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_MACD",
        "description": "Calculate the Moving Average Convergence Divergence (MACD) for the given stock ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., 'AAPL')"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate the Relative Strength Index (RSI) for the given stock ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., 'AAPL')"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "plot_stock_price",
        "description": "Plot the stock price for the given ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., AAPL for Apple)"
                }
            },
            "required": ["ticker"]
        }
    }
]


available_functions = {
    "get_stock_price": get_stock_price,
    "calculate_SMA": calculate_SMA,
    "calculate_EMA": calculate_EMA,
    "calculate_MACD": calculate_MACD,
    "calculate_RSI": calculate_RSI,
    "plot_stock_price": plot_stock_price

}

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Stock Analysis ChatBot Assistant")

user_input = st.text_input("Enter your message:")

if user_input:
    try:
        st.session_state["messages"].append({"role": "user", "content": f"{user_input}"})  # add user message to the conversation

        
        response= openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106", 
            messages=st.session_state["messages"],
            functions= functions,
            functions_call = "auto" ) # get response from GPT-3
        
        response_message = response.choices[0]["message"] # get response message from GPT-3

        if response_message.get("funcion_call"):
            function_name = response_message["function_call"]["name"] # get function name 
            function_args = json.loads(response_message["function_call"]["arguments"]) # get function arguments as JSON
            if function_name in ["get_stock_price", "calculate_RSI", "calculate_MACD", "plot_stock_price"]:
                args_dict = {"ticker": function_args.get("ticker")} # get ticker from function arguments
            elif function_name in ["calculate_SMA", "calculate_EMA"]:
                args_dict = {"ticker": function_args.get("ticker"), "window": function_args.get("window")} # get ticker from function arguments           
            
            function_to_call = available_functions[function_name] # get function to call from available functions
            function_response = function_to_call(**args_dict)

            if function_name == "plot_stock_price":
                st.image("stock_price.png")
            else:
                st.session_state["messages"].append(response_message) # add response message to the conversation
                st.session_state["messages"].append(
                    {"role": "assistant",
                     "name": function_name, 
                     "content": f"{function_response}"}
                     ) # add function response to the conversation
                
                second_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=st.session_state["messages"],
                    
                )
                st.text(second_response["choices"][0]["message"]["content"]) # display second response message from GPT-3 # generate a new prompt with assistant's answer
                st.session_state["messages"].append({"role": "assistant", "content": second_response.choices[0]["message"]["content"]}) # add second response message to the conversation  # ask for another question or continue the conversation
    
        else:
            st.text(response_message["content"]) # display response message from GPT-3 # generate a new prompt with assistant's answer
            st.session_state["messages"].append({"role": "assistant", "content": response_message["content"]}) # add response message to the conversation # show assistant's response # display assistant's response
    except openai.OpenAIError as e:
      st.error(f"OpenAI ile ilgili bir hata oluştu: {e}") # OpenAI hatalarını yakalayın ve kullanıcıya bildirin
    except yf.YFinanceError as e:
        st.error(f"YFinance ile ilgili bir hata oluştu: {e}") # YFinance hatalarını yakalayın ve kullanıcıya bildirin
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}") # Diğer tüm hataları yakalayın ve kullanıcıya bildirin


