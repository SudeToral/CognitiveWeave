./scripts/server.sh ps       # container durumu
./scripts/server.sh deploy   # git pull
./scripts/server.sh restart  # pull + yeniden başlat
./scripts/server.sh logs     # son loglar

deploy → Kod değişikliği yaptın, server'daki repo'yu güncellemek istiyorsun
restart → Hem güncelle hem container'ları yeniden başlat
ps → "Container'lar ayakta mı?" diye kontrol et
logs → Bir şeyler yanlış gittiğinde Neo4j veya Redis loglarına b