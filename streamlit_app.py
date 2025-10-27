
Controls

Data auto-refreshes every 5 minutes

📈 Stock & Crypto Performance Dashboard (AI-Powered)
KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/sheep-count/streamlit_app.py", line 145, in <module>
    stock_data, crypto_data = fetch_all_data()
                              ~~~~~~~~~~~~~~^^
File "/mount/src/sheep-count/streamlit_app.py", line 111, in fetch_all_data
    crypto_data[c] = f.result()
                     ~~~~~~~~^^
File "/usr/local/lib/python3.13/concurrent/futures/_base.py", line 449, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
File "/usr/local/lib/python3.13/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
File "/usr/local/lib/python3.13/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 227, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 269, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 328, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
File "/mount/src/sheep-count/streamlit_app.py", line 86, in get_crypto_data
    prices = pd.DataFrame(data["prices"], columns=["Date", "Price"])
                          ~~~~^^^^^^^^^^
