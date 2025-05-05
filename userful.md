## ollamani portini chiqarish:

- windows: 
```
set OLLAMA_HOST=0.0.0.0
ollama serve
```

- linux:
```
OLLAMA_HOST=0.0.0.0 ollama serve
```


- open ip port outside from local
```
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=11434 connectaddress=127.0.0.1 connectport=11434
```

- close ip port outside from local
```
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=11434
```