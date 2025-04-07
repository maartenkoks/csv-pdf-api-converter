import openai
openai.api_key = "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"
models = openai.Model.list()
for m in models['data']:
    print(m['id'])