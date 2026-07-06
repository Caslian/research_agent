"""
用户相关API路由

MVP 模式：通过 email 落地（不接 JWT/login）——前端用一个固定 demo 邮箱，
后端若该 email 不存在则创建。这样 KB / chat 等需要 user_id 的接口能直接跑通。
后续上 JWT 时把 `get_or_create_user` 替换为 token 验证即可。
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging
import uuid
from core.database import db_manager
from core.exceptions import DatabaseException

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic模型
class UserCreateRequest(BaseModel):
    email: str
    profile: Optional[Dict[str, Any]] = {}


class UserUpdateRequest(BaseModel):
    profile: Optional[Dict[str, Any]] = {}

class UserResponse(BaseModel):
    id: str
    email: str
    profile: Dict[str, Any]
    created_at: str

@router.post("/", response_model=UserResponse)
async def create_user(request: UserCreateRequest):
    """创建用户（MVP：email 唯一约束，重复插入将被 except 拦截；建议先用 /by-email）"""
    try:
        try:
            user_id = await db_manager.create_user(
                email=request.email,
                profile=request.profile
            )
        except DatabaseException as e:
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                existing = await db_manager.fetchrow(
                    "SELECT id FROM users WHERE email = $1", request.email,
                )
                if existing:
                    user_id = str(existing["id"])
                else:
                    raise
            else:
                raise

        user = await db_manager.get_user(str(user_id))
        return UserResponse(
            id=str(user["id"]),
            email=user["email"],
            profile=user.get("profile") or {},
            created_at=user["created_at"].isoformat() if user.get("created_at") else ""
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建用户失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-email/{email}", response_model=UserResponse)
async def get_or_create_user_by_email(email: str):
    """MVP：通过 email 取 user，不存在则自动创建。前端 KB 初始化用这个。"""
    try:
        existing = await db_manager.fetchrow(
            "SELECT id FROM users WHERE email = $1", email,
        )
        if existing:
            user_id = str(existing["id"])
        else:
            user_id = await db_manager.create_user(email=email, profile={"name": email})
        user = await db_manager.get_user(user_id)
        return UserResponse(
            id=str(user["id"]),
            email=user["email"],
            profile=user.get("profile") or {},
            created_at=user["created_at"].isoformat() if user.get("created_at") else ""
        )
    except Exception as e:
        logger.error(f"get_or_create 用户失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """获取用户信息"""
    try:
        user = await db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        
        return UserResponse(
            id=user["id"],
            email=user["email"],
            profile=user["profile"],
            created_at=user["created_at"].isoformat() if user["created_at"] else ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{user_id}", response_model=Dict[str, Any])
async def update_user(user_id: str, request: UserUpdateRequest):
    """更新用户信息"""
    try:
        success = await db_manager.update_user_profile(
            user_id=user_id,
            profile=request.profile
        )
        
        if success:
            return {"success": True, "message": "用户信息已更新"}
        else:
            raise HTTPException(status_code=404, detail="用户不存在")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}/profile")
async def get_user_profile(user_id: str):
    """获取用户配置"""
    try:
        user = await db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        
        return {
            "success": True,
            "profile": user.get("profile", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{user_id}/profile")
async def update_user_profile(user_id: str, profile: Dict[str, Any]):
    """更新用户配置"""
    try:
        success = await db_manager.update_user_profile(
            user_id=user_id,
            profile=profile
        )
        
        if success:
            return {"success": True, "message": "用户配置已更新"}
        else:
            raise HTTPException(status_code=404, detail="用户不存在")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))