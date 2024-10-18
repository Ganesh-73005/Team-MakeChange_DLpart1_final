
### Blog Application
```markdown


## Overview
This is a simple blog application built using Flask and MongoDB. Users can create and manage blog posts, and the application supports sharding to enhance scalability and performance.

## Technologies Used
- Python
- Flask
- MongoDB
- PyMongo

## Sharding


### Sharding Steps

To implement sharding, follow these steps:

1. **Enable Sharding on the Database**:
   ```bash
   use admin
   sh.enableSharding("blog_database")
   ```

2. **Shard the `blog_posts` Collection**:
   ```bash
   use blog_database
   sh.shardCollection("blog_database.blog_posts", { "date": hashed })
   ```

3. **Shard the `users` Collection with Hashed Sharding**:
   ```bash
   use blog_database
   sh.shardCollection("blog_database.users", { "_id": "hashed" })
   ```






## Overview
This is a Flask application that allows users to create and manage blog posts, comment on posts, and handle user authentication. The application uses MongoDB for data storage, managed through the `flask_pymongo` extension.

## Features
- User registration and login
- Create, read, update, and delete blog posts
- Comment on blog posts
- Search functionality for blog posts
- Secure password storage using hashing

## Technologies Used
- **Flask**: A micro web framework for Python.
- **MongoDB**: A NoSQL database for storing application data.
- **flask_pymongo**: Extension for integrating Flask with MongoDB.
- **HTML/CSS**: For front-end development.
- **Bootstrap**: For responsive design.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flask-blog-app.git
   cd flask-blog-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up MongoDB:
   - Ensure MongoDB is installed and running on your machine.
   - Create a database for the application.

5. Update the configuration in `config.py`:
   ```python
   MONGO_URI = "mongodb://localhost:27017/yourdatabase"
   ```

6. Run the application:
   ```bash
   flask run
   ```

## MongoDB Schemas

### 1. Users Collection
```json
{
    "_id": "ObjectId",                    
    "email": "string",                    
    "password": "string",                 
    "name": "string"                      
}
```

### 2. Blog Posts Collection
```json
{
    "_id": "ObjectId",                    
    "title": "string",                    
    "subtitle": "string",                 
    "body": "string",                     
    "img_url": "string",                  
    "author": "string",                   
    "date": "string",                     
    "timestamp": "string"                 
}
```

### 3. Blog Comments Collection
```json
{
    "_id": "ObjectId",                    
    "text": "string",                     
    "comment_author": "string",           
    "parent_post": "ObjectId"             
}
```

## MongoDB Queries Used

### Reading Data (GET)
- **Get all blog posts**:
   ```python
   posts = list(mongo.db.blog_posts.find())
   ```

- **Get a specific post by ID**:
   ```python
   requested_post = mongo.db.blog_posts.find_one({"_id": ObjectId(post_id)})
   ```

- **Get all comments for a specific post**:
   ```python
   requested_post_comments = mongo.db.blog_comments.find({"parent_post": ObjectId(post_id)})
   ```

### Creating Data (POST)
- **Insert a new user**:
   ```python
   mongo.db.users.insert_one(new_user)
   ```

- **Insert a new blog post**:
   ```python
   mongo.db.blog_posts.insert_one(new_post)
   ```

- **Insert a new comment**:
   ```python
   mongo.db.blog_comments.insert_one(new_comment)
   ```

### Updating Data (PUT/PATCH)
- **Update an existing blog post**:
   ```python
   mongo.db.blog_posts.update({"_id": ObjectId(post_id)}, post)
   ```

### Deleting Data (DELETE)
- **Delete a blog post by ID**:
   ```python
   mongo.db.blog_posts.remove({"_id": ObjectId(post_id)})
   ```

- **Delete a comment by ID**:
   ```python
   mongo.db.blog_comments.remove({"_id": ObjectId(comment_id)})
   ```

## Summary
This project demonstrates how to effectively utilize MongoDB with Flask to build a fully functional blog application. It supports CRUD operations for users, blog posts, and comments, providing a comprehensive user experience.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [flask_pymongo Documentation](https://flask-pymongo.readthedocs.io/en/latest/)
```

### Instructions
- Replace `https://github.com/yourusername/flask-blog-app.git` with the actual repository link.
- Modify any sections as necessary to match your project’s specifics.
- Ensure to include any additional installation steps or features that may be relevant.
