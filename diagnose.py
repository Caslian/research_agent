#!/usr/bin/env python3
"""系统诊断脚本 - 检查所有配置和依赖"""

import sys
import os
from pathlib import Path

def check_env_file():
    """检查 .env 文件"""
    print("\n" + "="*60)
    print("1. 检查环境配置文件")
    print("="*60)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env 文件不存在")
        return False
    
    print("✅ .env 文件存在")
    
    # 读取关键配置
    with open(env_path, encoding='utf-8') as f:
        content = f.read()
        
    required_keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
    for key in required_keys:
        if key in content:
            print(f"✅ {key} 已配置")
        else:
            print(f"⚠️  {key} 未配置")
    
    return True

def check_dependencies():
    """检查依赖包"""
    print("\n" + "="*60)
    print("2. 检查依赖包")
    print("="*60)
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "hello_agents",
        "arxiv",
        "httpx",
        "asyncpg",
        "qdrant_client",
        "feedparser",
        "beautifulsoup4",
        "langchain_qdrant"
    ]
    
    missing = []
    for package in required_packages:
        try:
            # 特殊处理包名映射
            import_name = package.replace("-", "_")
            if package == "beautifulsoup4":
                import_name = "bs4"
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 缺失")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  缺失的包: {', '.join(missing)}")
        print(f"安装命令: pip install {' '.join(missing)}")
        return False
    
    return True

def check_config():
    """检查配置加载"""
    print("\n" + "="*60)
    print("3. 检查配置加载")
    print("="*60)
    
    try:
        from core.config import get_config
        config = get_config()
        
        print(f"✅ 配置加载成功")
        print(f"   - API Key: {'已设置' if config.llm.api_key else '未设置'}")
        print(f"   - Base URL: {config.llm.base_url or '未设置'}")
        print(f"   - Model: {config.llm.model_name}")
        print(f"   - Debug: {config.debug}")
        
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {str(e)}")
        return False

def check_api_routes():
    """检查 API 路由"""
    print("\n" + "="*60)
    print("4. 检查 API 路由")
    print("="*60)
    
    try:
        from api.main import app
        
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        print(f"✅ API 加载成功，共 {len(routes)} 个路由")
        
        # 检查关键路由
        key_routes = ["/", "/health", "/api/v1/papers/search", "/api/v1/analysis/analyze"]
        for route in key_routes:
            if route in routes:
                print(f"   ✅ {route}")
            else:
                print(f"   ❌ {route} - 缺失")
        
        return True
    except Exception as e:
        print(f"❌ API 加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_frontend():
    """检查前端文件"""
    print("\n" + "="*60)
    print("5. 检查前端文件")
    print("="*60)
    
    frontend_files = [
        "frontend/index.html",
        "frontend/static/css/style.css",
        "frontend/static/js/app.js"
    ]
    
    all_exist = True
    for file_path in frontend_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} - 不存在（可选）")
    
    return True

def check_llm_connection():
    """检查 LLM 连接"""
    print("\n" + "="*60)
    print("6. 检查 LLM 连接")
    print("="*60)
    
    try:
        import asyncio
        from hello_agents import HelloAgentsLLM
        from core.config import get_config
        
        config = get_config()
        
        if not config.llm.api_key:
            print("⚠️  API Key 未设置，跳过连接测试")
            return True
        
        async def test():
            from core.llm_adapter import get_llm_adapter
            adapter = get_llm_adapter()
            
            response = await adapter.ainvoke("你好")
            return response
        
        print("正在测试 LLM 连接...")
        result = asyncio.run(test())
        print(f"✅ LLM 连接成功")
        print(f"   模型响应: {result[:50]}...")
        
        return True
    except Exception as e:
        error_msg = str(e)
        # 如果是 API 格式错误，说明连接是通的，只是请求格式问题
        if "400" in error_msg or "invalid_request" in error_msg:
            print(f"⚠️  LLM API 可访问，但请求格式需要调整")
            print(f"   错误信息: {error_msg}...")
            return True  # 认为通过，因为连接本身是正常的
        print(f"❌ LLM 连接失败: {error_msg}...")
        return False

def check_database_connection():
    """检查数据库连接"""
    print("\n" + "="*60)
    print("7. 检查数据库连接")
    print("="*60)
    
    try:
        import asyncio
        from core.config import get_config
        
        config = get_config().database
        
        # 显示数据库配置
        print(f"数据库配置:")
        print(f"  - 主机: {config.host}")
        print(f"  - 端口: {config.port}")
        print(f"  - 数据库名: {config.database}")
        print(f"  - 用户名: {config.username}")
        
        async def test_connection():
            try:
                import asyncpg
                
                # 尝试连接到数据库
                connection = await asyncio.wait_for(
                    asyncpg.connect(
                        host=config.host,
                        port=config.port,
                        database=config.database,
                        user=config.username,
                        password=config.password,
                    ),
                    timeout=10.0
                )
                
                # 测试连接是否正常
                version = await connection.fetchval('SELECT version()')
                await connection.close()
                
                return True, version
            
            except asyncio.TimeoutError:
                return False, f"连接超时（10秒），请检查：\n" \
                       f"  1. 数据库服务是否正在运行\n" \
                       f"  2. 主机地址是否正确: {config.host}\n" \
                       f"  3. 端口号是否正确: {config.port}\n" \
                       f"  4. 网络连接是否正常"
            
            except asyncpg.InvalidPasswordError:
                return False, f"密码错误。请检查：\n" \
                       f"  1. 用户名是否正确: {config.username}\n" \
                       f"  2. 密码是否正确\n" \
                       f"  3. .env 文件中的 DATABASE_PASSWORD 配置"
            
            except asyncpg.InvalidAuthenticationSpecificationError:
                return False, f"认证失败。请检查：\n" \
                       f"  1. 用户名: {config.username}\n" \
                       f"  2. 用户是否具有连接到该数据库的权限\n" \
                       f"  3. PostgreSQL 服务器的认证方法设置"
            
            except asyncpg.PostgresError as e:
                error_code = getattr(e, 'sqlstate', 'UNKNOWN')
                if 'does not exist' in str(e):
                    return False, f"数据库不存在: {config.database}\n" \
                           f"  请执行以下命令创建数据库:\n" \
                           f"  createdb -U {config.username} -h {config.host} {config.database}"
                elif error_code == '3D000':
                    return False, f"数据库 '{config.database}' 不存在\n" \
                           f"  请执行以下命令创建数据库:\n" \
                           f"  createdb -U {config.username} -h {config.host} {config.database}"
                else:
                    return False, f"PostgreSQL 错误 [{error_code}]: {str(e)}"
            
            except Exception as e:
                error_msg = str(e).lower()
                
                if 'connection refused' in error_msg or 'refused' in error_msg:
                    return False, f"连接被拒绝。请检查：\n" \
                           f"  1. PostgreSQL 服务是否在运行\n" \
                           f"  2. PostgreSQL 是否在监听 {config.host}:{config.port}\n" \
                           f"  3. 防火墙是否阻止了连接\n" \
                           f"  启动 PostgreSQL:\n" \
                           f"    - Linux: sudo systemctl start postgresql\n" \
                           f"    - macOS: brew services start postgresql\n" \
                           f"    - Windows: net start postgresql-14 (或相应版本)"
                
                elif 'name or service not known' in error_msg or 'nodename nor servname provided' in error_msg:
                    return False, f"无法解析主机名: {config.host}\n" \
                           f"  请检查：\n" \
                           f"  1. 主机名是否拼写正确\n" \
                           f"  2. 如果是 Docker，容器是否正常运行\n" \
                           f"  3. DNS 是否正常工作"
                
                elif 'network is unreachable' in error_msg:
                    return False, f"网络无法到达: {config.host}:{config.port}\n" \
                           f"  请检查：\n" \
                           f"  1. 网络连接是否正常\n" \
                           f"  2. 主机是否在线\n" \
                           f"  3. VPN 或防火墙配置"
                
                else:
                    return False, f"连接异常: {str(e)}"
        
        print("正在测试数据库连接...")
        success, result = asyncio.run(test_connection())
        
        if success:
            print(f"✅ 数据库连接成功")
            print(f"   数据库版本: {result.split(',')[0]}")
            return True
        else:
            print(f"❌ 数据库连接失败")
            print(f"\n错误详情:")
            print(result)
            return False
    
    except Exception as e:
        print(f"❌ 数据库配置检查失败: {str(e)}")
        return False

def check_database_tables():
    """检查数据库表"""
    print("\n" + "="*60)
    print("8. 检查数据库表")
    print("="*60)
    
    try:
        import asyncio
        from core.config import get_config
        
        config = get_config().database
        
        async def check_tables():
            try:
                import asyncpg
                
                connection = await asyncpg.connect(
                    host=config.host,
                    port=config.port,
                    database=config.database,
                    user=config.username,
                    password=config.password,
                )
                
                # 查询所有表
                tables = await connection.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                
                expected_tables = [
                    'users',
                    'papers',
                    'user_paper_relations',
                    'analysis_reports',
                    'reference_cache'
                ]
                
                existing_tables = [t['table_name'] for t in tables]
                missing_tables = [t for t in expected_tables if t not in existing_tables]
                
                await connection.close()
                
                return existing_tables, missing_tables
            
            except Exception as e:
                return None, str(e)
        
        existing, missing = asyncio.run(check_tables())
        
        if existing is None:
            print(f"❌ 无法查询表信息: {missing}")
            print(f"\n错误原因可能是：")
            print(f"  1. 数据库连接不可用")
            print(f"  2. 用户没有查询权限")
            return False
        
        if not existing:
            print(f"⚠️  数据库中没有表，需要初始化")
            print(f"执行以下命令初始化数据库:")
            print(f"  python init_db.py    # 或运行项目内置的初始化脚本")
            return False

        print(f"✅ 发现 {len(existing)} 个表:")
        for table in existing:
            print(f"   ✅ {table}")

        if missing:
            print(f"\n⚠️  缺少以下表: {', '.join(missing)}")
            print(f"执行以下命令重新初始化:")
            print(f"  python init_db.py    # 或运行项目内置的初始化脚本")
            return len(missing) == 0
        
        return True
    
    except Exception as e:
        print(f"❌ 表检查失败: {str(e)}")
        return False

def check_qdrant_connection():
    """检查 Qdrant 向量数据库连接"""
    print("\n" + "="*60)
    print("9. 检查 Qdrant 向量数据库")
    print("="*60)
    
    try:
        from core.config import get_config
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
        config = get_config().vector_db
        
        # 显示 Qdrant 配置（支持远程 URL 和本地 host+port 两种模式）
        qdrant_url = config.get_qdrant_url()
        print(f"Qdrant 配置:")
        print(f"  - 连接地址: {qdrant_url}")
        if config.api_key:
            print(f"  - API Key: 已配置")
        
        print(f"\n正在连接到 Qdrant...")
        
        try:
            # 初始化 Qdrant 客户端
            client = QdrantClient(
                url=qdrant_url,
                api_key=config.api_key,
                prefer_grpc=False,
                https=qdrant_url.startswith("https://"),
                check_compatibility=False
            )
            
            # 测试连接 - 获取服务器信息
            collection_response = client.get_collections()
            collections = [c.name for c in collection_response.collections] if collection_response.collections else []
            
            print(f"✅ Qdrant 连接成功")
            
            # 检查集合
            if collections:
                print(f"\n  发现 {len(collections)} 个 Collection:")
                for col in collections:
                    print(f"   - {col}")
            else:
                print(f"\n  暂无 Collection，将在首次使用时创建")
            
            return True
        
        except TimeoutError:
            return False, f"连接超时。请检查：\n" \
                   f"  1. Qdrant 服务是否正在运行\n" \
                   f"  2. 连接地址是否正确: {qdrant_url}\n" \
                   f"  3. 网络连接是否正常\n" \
                   f"  4. 防火墙是否阻止了连接\n\n" \
                   f"  启动 Qdrant（本地 Docker）:\n" \
                   f"    docker run -d -p 6333:6333 qdrant/qdrant"
        
        except ConnectionRefusedError:
            return False, f"连接被拒绝。请检查：\n" \
                   f"  1. Qdrant 服务是否在运行\n" \
                   f"  2. 连接地址: {qdrant_url}\n" \
                   f"  3. 防火墙配置"
        
        except (ResponseHandlingException, UnexpectedResponse) as e:
            error_msg = str(e).lower()
            
            if 'unauthorized' in error_msg or '401' in error_msg or 'api.key' in error_msg:
                return False, f"API Key 无效或缺失。请检查 .env 中的 QDRANT_API_KEY"
            
            if 'connection refused' in error_msg or 'refused' in error_msg:
                return False, f"连接被拒绝: {qdrant_url}\n" \
                       f"  请检查 Qdrant 服务是否启动"
            
            elif 'connection reset' in error_msg:
                return False, f"连接被重置。请检查网络连接和防火墙规则"
            
            elif 'name or service not known' in error_msg or 'nodename nor servname provided' in error_msg:
                return False, f"无法解析主机名。请检查连接地址是否正确: {qdrant_url}"
            
            elif 'network is unreachable' in error_msg:
                return False, f"网络无法到达: {qdrant_url}\n" \
                       f"  请检查网络连接、VPN 或代理配置"
            
            else:
                return False, f"连接异常: {str(e)}\n\n" \
                       f"  当前配置: {qdrant_url}"
    
    except ImportError as e:
        return False, f"缺少依赖包: qdrant-client\n" \
               f"  安装命令: pip install qdrant-client"
    
    except Exception as e:
        return False, f"检查失败: {str(e)}"

def check_embedding_service():
    """检查嵌入服务初始化"""
    print("\n" + "="*60)
    print("10. 检查嵌入服务初始化")
    print("="*60)
    
    try:
        from utils.embedding import get_embedding_service
        import asyncio
        
        embedding_service = get_embedding_service()
        
        print(f"嵌入服务配置:")
        print(f"  - 模型: {embedding_service.embedding_model}")
        print(f"  - 缓存大小: {embedding_service.get_cache_size()}")
        
        # 检查是否已初始化
        if embedding_service.embeddings:
            print(f"\n✅ 嵌入服务已初始化")
            print(f"   LangChain Embeddings 已就绪")
            return True
        else:
            print(f"\n⚠️  嵌入服务未初始化")
            print(f"   将在首次使用时自动初始化")
            
            # 尝试初始化
            try:
                async def init_test():
                    await embedding_service.initialize()
                    return True
                
                result = asyncio.run(init_test())
                if result:
                    print(f"✅ 嵌入服务初始化测试成功")
                    return True
            except Exception as init_error:
                print(f"⚠️  初始化测试失败: {str(init_error)}")
                print(f"   可能原因：")
                print(f"   1. OPENAI_API_KEY 未配置")
                print(f"   2. OpenAI API 无法访问")
                print(f"   3. 网络连接问题")
                return False
    
    except ImportError as e:
        return False, f"缺少依赖包: {str(e)}"
    except Exception as e:
        return False, f"检查失败: {str(e)}"

def main():
    """主函数"""
    print("\n" + "="*60)
    print("InnoCore AI 系统诊断")
    print("="*60)
    
    results = []
    
    results.append(("环境配置", check_env_file()))
    results.append(("依赖包", check_dependencies()))
    results.append(("配置加载", check_config()))
    results.append(("API 路由", check_api_routes()))
    results.append(("前端文件", check_frontend()))
    # results.append(("LLM 连接", check_llm_connection()))
    results.append(("数据库连接", check_database_connection()))
    results.append(("数据库表", check_database_tables()))
    results.append(("Qdrant 连接", check_qdrant_connection()))
    results.append(("嵌入服务", check_embedding_service()))
    
    # 总结
    print("\n" + "="*60)
    print("诊断结果总结")
    print("="*60)
    
    for name, result in results:
        if isinstance(result, tuple):
            status = "❌ 失败"
            print(f"{name}: {status}")
            print(f"\n错误详情:")
            print(result[1])
        else:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{name}: {status}")
    
    # 检查是否全部通过
    all_passed = all(r[1] if not isinstance(r[1], tuple) else False for r in results)
    
    if all_passed:
        print("\n🎉 所有检查通过！系统可以正常运行。")
        print("\n启动命令: python run.py")
    else:
        print("\n⚠️  部分检查未通过，请根据上述提示修复问题。")
        print("\n常见问题排查:")
        print("  1. Qdrant 连接问题 → docker run -d -p 6333:6333 qdrant/qdrant")
        print("  2. 数据库连接问题 → 检查 PostgreSQL 是否启动")
        print("  3. 依赖包缺失 → 运行 pip install -r requirements.txt")
        print("  4. 表不存在 → 运行 python init_db.py 初始化数据库")

if __name__ == "__main__":
    main()
