IMAGE_VERSION:=v$(shell date +"%Y%m%d%H%M")

DOCKER_FILE:= Dockerfile
IMAGE_NAME:=ubuntu/python-example/gpu-server:$(IMAGE_VERSION)
docker: docker-active
	@docker buildx build -t "$(IMAGE_NAME)" -f "$(DOCKER_FILE)" .
	@docker builder prune -a -f
	@echo "Image Name: $(IMAGE_NAME)"
	@echo "done"
.PHONY: docker

docker-active:
	@if ! command -v docker &> /dev/null ; then \
         echo "Docker 未安裝"; \
         exit 1; \
    fi
	@if ! systemctl is-active --quiet docker ; then \
		echo "Docker 未启动"; \
		exit 2; \
	fi
	@if docker info &> /dev/null ; then \
		echo "Docker 可用"; \
	else \
		echo "Docker 不可用,可能原因:权限配置有问题"; \
		exit 3; \
	fi
.PHONY: docker-active

clean: docker-active
	@docker builder prune -a -f
	@echo "done"
.PHONY: clean
