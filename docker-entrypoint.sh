#!/bin/bash


user=appuser

# 切换到指定用户重新执行 shell
if [ "$(id -u)" -eq 0 ]; then
  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] change user to $user"
  exec gosu "$user" "$0" "$@"
fi

function printHelp(){
  echo ""
  echo "运行django的帮助信息"
  echo ""
  echo "Commands:"
  echo "  migrate: 只做迁移命令,会自动执行makemigrations"
  echo "  runserver: 运行 python manege.py runserver 服务命令, 必选 -address 的 Option"
  echo "  serve: 运行 gunicorn 服务命令 可选 -address 的 Option, 必选 -config 的 Option"
  echo ""
  echo "Options:"
  echo "  -migrate 同时需要执行迁移命名,等同于 migrate"
  echo "  -address HOST:PORT 配置运行服务地址"
  echo "  -config gunicorn命令的配置文件"
  echo ""
  echo "Example:"
  echo "  runserver -address 0.0.0.0:8000 -migrate"
  echo "  migrate"
  echo "  serve -address 0.0.0.0:8080 -migrate -config config/gunicorn.config.py"
  echo ""
}


# 运行命令
MODE=""

if [[ $# -lt 1 ]]; then
  echo "$(date +"%Y-%m-%d %H:%M:%S") [WARN ] 缺少运行命令"
  printHelp
  exit 1
else
  MODE="$1"
  shift
fi

while [[ $# -ge 1 ]]; do
  key="$1"
  case $key in
   -migrate )
     MIGRATE="1"
     #shift
     ;;
   -address )
     if [[ -z "$2" ]]; then
       echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] 缺少运行地址信息"
       printHelp
       exit 1
     fi
     ADDRESS="$2"
     shift
     ;;
   -config )
     if [[ -z "$2" ]]; then
       echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] 缺少gunicorn的配置文件信息"
       printHelp
       exit 1
    fi
    GUNICORN_CONFIG="$2"
    shift
    ;;
   * )
     echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] 未知Option: $key"
     printHelp
     exit 1
     ;;
   esac
   shift
done


CUDA_ENABLE=$(python -c "import torch;print(torch.cuda.is_available())")

echo ""
echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] use user:{$(whoami)} run app with cuda: {$CUDA_ENABLE}"
echo ""


function command_migrate() {
  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] 正在运行 python manage.py makemigrations..."
  python manage.py makemigrations
  echo ""
  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] 正在运行 python manage.py migrate..."
  python manage.py migrate
  echo ""
}


if [[ "$MODE" == "migrate" ]]; then
  command_migrate
elif [[ "$MODE" == "runserver" ]];then
  # 判断是否有监听地址
  if [[ -z "$ADDRESS" ]]; then
    echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] 运行服务缺少监听地址"
    printHelp
    exit 1
  fi

  # 到这认为有监听地址
  if [[ "$MIGRATE" == "1" ]]; then
    command_migrate
  fi

  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] 运行 python manage.py runserver $ADDRESS"
  exec python manage.py runserver "$ADDRESS"
elif [[ "$MODE" == "serve" ]]; then

  if [[ -z "$GUNICORN_CONFIG" ]]; then
    echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] 缺少gunicorn配置文件信息"
    printHelp
    exit 1
  fi

  if [[ ! -f "$GUNICORN_CONFIG" ]]; then
    echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] Gunicorn 配置文件不存在: $GUNICORN_CONFIG"
    exit 1
  fi

  if [[ "$MIGRATE" == "1" ]]; then
    command_migrate
  fi

  if [[ -z "$ADDRESS" ]]; then
    echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] 运行 gunicorn --config $GUNICORN_CONFIG"
    exec gunicorn --config $GUNICORN_CONFIG
  else
    echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO ] 运行 gunicorn --bind $ADDRESS --config $GUNICORN_CONFIG"
    exec gunicorn --bind $ADDRESS --config $GUNICORN_CONFIG
  fi

else
  echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] 未知命令: $MODE"
  printHelp
  exit 1
fi
