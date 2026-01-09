import streamlit as st
from helpers import *
import os

def main():
    # que="Python with SSMS"
    # resp=serp_req(que)
    # print(resp)
    # urls=urls_picker(resp,que)
    # print(urls)
    # data=extract_content(urls=urls)
    # summary=summarizer(data,que)
    # print(summary)
    # newsletter=generate_newsletter(summary,que)
    # print(newsletter)
    st.set_page_config(page_title="NewsLetter Generator",page_icon=":parrot:",layout="wide")

    st.header("Generate a NewsLetter :parrot:")
    query=st.text_input("Enter a topic to generate a newsletter:")

    if query:
        print(query)
        with st.spinner(f"Generating NewsLetter for the provided {query}"):
            search_results=serp_req(query)
            urls=urls_picker(search_results,query)
            extract_data=extract_content(urls)
            summaries=summarizer(extract_data,query)
            result=generate_newsletter(summaries,query)

            with st.expander("Search Results:"):
                st.info(search_results)
            with st.expander("Top URLS"):
                st.info(urls)
            with st.expander("Data"):
                raw_data=" ".join(d.page_content for d in extract_data.similarity_search(query, k=4))
                st.info(extract_data)
            with st.expander("Summary of the data extracted:"):
                st.info(summaries)
            with st.expander("NewsLetter"):
                st.info(result)
        st.success("Done!")

if __name__ =="__main__":
    main()