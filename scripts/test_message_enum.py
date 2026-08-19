from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app import models
from app.database import Base

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
db = Session()

user = models.User(
    username="enum-test",
    email="enum-test@example.com",
    hashed_password="not-used",
)
db.add(user)
db.flush()
project = models.Project(name="Enum Test", created_by=user.id)
db.add(project)
db.flush()
conversation = models.Conversation(project_id=project.id, user_id=user.id)
db.add(conversation)
db.flush()
db.add(models.Message(
    conversation_id=conversation.id,
    role=models.MessageRole.USER,
    content="test",
))
db.commit()

with engine.connect() as connection:
    stored_role = connection.execute(
        text("SELECT role FROM messages WHERE content = 'test' LIMIT 1")
    ).scalar_one()

assert stored_role == "user", stored_role
assert db.query(models.Message).one().role is models.MessageRole.USER
print("message_enum_flow=passed")
