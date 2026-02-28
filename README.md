# PythohFlaskEmbededAPI
python api werser generate text to embeded 768 array

# Run now 
```
docker run -d -p 5000:5000 --name embed passapol/pythonflaskembeded-api:v1
```

# Build Docker
```
docker build -t pythonflaskembeded-api .
docker run -p 5000:5000 pythonflaskembeded-api
```

# Test API
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "สวัสดีครับ"}'

curl -X POST http://localhost:5000/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello", "สวัสดี"]}'
