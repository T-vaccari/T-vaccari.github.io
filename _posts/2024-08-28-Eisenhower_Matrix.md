---
layout: post
title: "Building a Simple Web App with Firebase Authentication and Firestore: A Step-by-Step Guide"
categories: jekyll update
---

In today’s post, I’ll walk you through the development of a simple web application, showcasing how to leverage Firebase Authentication and Firestore Database. This guide will help you understand how to:

1. Authenticate users with Firebase Authentication.
2. Store and retrieve data in real time using Firestore.

To make things more practical, we’ll build an application based on the [Eisenhower Matrix](https://en.wikipedia.org/wiki/Time_management#The_Eisenhower_Method), a powerful decision-making framework that helps you prioritize tasks based on urgency and importance.

If you want to see the final result in action, feel free to explore the app [here](https://eisenhower-matrix.tommasovaccari.com).

## What is the Eisenhower Matrix?

The Eisenhower Matrix is divided into four quadrants:

- **Quadrant 1**: Urgent and Important
- **Quadrant 2**: Not Urgent but Important
- **Quadrant 3**: Urgent but Not Important
- **Quadrant 4**: Neither Urgent nor Important

This framework helps you classify tasks and make better decisions about what to prioritize, what to plan for later, and what to delegate or remove.

## Overview of the App

My web application brings the Eisenhower Matrix to life allowing users to:

- **Create an account** using Firebase Authentication.
- **Add tasks** and assign them to one of the four quadrants (Urgent/Important, Not Urgent/Important, Urgent/Not Important, Not Urgent/Not Important).
- **Move tasks dynamically** between quadrants as priorities change.
- **Store tasks** securely in the cloud using Firestore, ensuring persistence across sessions and accessibility from any device.

### Core Components

- **Firebase Authentication**: Enables users to create accounts, log in, and secure access to their personalized task data.
- **Firestore Database**: A NoSQL cloud database that stores user tasks in real time, ensuring that changes are immediately reflected in the UI and synced across devices.
- **HTML, JavaScript, and CSS**: The front-end is built using standard web technologies. HTML structures the layout, JavaScript provides interactivity (e.g., adding and moving tasks), and CSS ensures the app is visually appealing and user-friendly.

## Why Firebase?

Firebase provides a robust suite of tools, making it ideal for small projects like this. It offers both authentication and database features, allowing you to manage user sessions and store data without having to set up complex backend infrastructure. The **free tier** is more than enough for small projects, allowing you to build and launch an app without any initial cost.

## Step-by-Step Guide

Now, let's dive into the app construction, step by step.

### Step 1: Setting Up Firebase

1. Go to the [Firebase Console](https://console.firebase.google.com) and create a new project.
2. Enable **Firebase Authentication**: Navigate to the "Authentication" tab, click "Set up sign-in method", and enable **Email/Password** authentication.
3. Enable **Firestore Database**: In the "Firestore Database" section, click "Create database", and choose the appropriate settings for your app (test mode is fine for development).

Now you're ready to integrate Firebase into your app!


### Step 2 : Integrate Firebase into the Web App

Once you've completed the basic structure and styling of your web application using HTML and CSS, you can now integrate Firebase services to handle authentication and data storage.

To do this, you need to import the necessary Firebase modules directly into your JavaScript file. Firebase provides a modern way to handle individual services using ES modules, which makes your code more modular and lightweight.

```javascript
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.0.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/9.0.0/firebase-auth.js";
import { getFirestore, doc, setDoc, collection, addDoc, onSnapshot, deleteDoc, query, where, limit, startAfter } from "https://www.gstatic.com/firebasejs/9.0.0/firebase-firestore.js";

```

- initializeApp: This function is essential to start using Firebase in your app. It initializes Firebase using your project’s configuration details.

Authentication Services:

- getAuth: Initializes Firebase Authentication, which is used to manage users and their login status..
- signInWithEmailAndPassword: Allows users to log in using their email and password.
- createUserWithEmailAndPassword: Enables new user registration with email and password.
- onAuthStateChanged: Listens to changes in authentication state, allowing you to track whether a user is logged in or logged out.
- signOut: Provides a method to log users out of the application.

Firestore Services:

- getFirestore: Initializes Firestore, the cloud database service that stores and syncs data in real time.
doc and setDoc: Used to create or reference specific documents in the Firestore database (for example, tasks in our Eisenhower Matrix app).
- collection and addDoc: Create or add data to a Firestore collection, such as adding a task to a user's list.
- onSnapshot: Listens to real-time changes in the database, so your app automatically reflects any updates made by the user.
- deleteDoc: Allows you to delete specific documents (e.g., remove tasks).
- query, where, limit, startAfter: Firestore query methods, which let you filter, paginate, and retrieve data efficiently from large collections.

By importing these modules, you only load the specific functionality you need for your web app, which is a more efficient approach, especially for smaller projects.

### Step 3 : Firebase Configuration and Initialization

Now you have to obtain the setting of your firebase project. You can find them in the setting section on the firebase console.

In your javascript file, initialize Firebase using the config setting :

```javascript
// Firebase config (replace with your own Firebase project settings)
const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_AUTH_DOMAIN",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_STORAGE_BUCKET",
    messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
    appId: "YOUR_APP_ID"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

```

### Step 4: Implement Firebase Authentication

Now that Firebase is initialized, let’s handle user authentication. We’ll add functionality for both signing up and logging in.
Now for the sake of clarity I use as an ID 'signUp' and 'logn' ,then you have to change it with the ID that you used in your HTML.

```javascript
// Sign up new users
document.getElementById('signUp').addEventListener('click', () => {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    auth.createUserWithEmailAndPassword(email, password)
        .then(userCredential => {
            alert("User signed up!");
        })
        .catch(error => {
            alert(error.message);
        });
});

// Log in existing users
document.getElementById('login').addEventListener('click', () => {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    auth.signInWithEmailAndPassword(email, password)
        .then(userCredential => {
            alert("User logged in!");
        })
        .catch(error => {
            alert(error.message);
        });
});

```

### Step 5: Firestore for Storing Tasks

Once the user is logged in, we’ll allow them to add and remove tasks to Firestore. Each task will be saved under the user’s unique ID in the Firestore database.

```javascript
// Add a task to Firestore
function addTask(quadrantId, task) {
    const userId = getCurrentUserId();
    if (!userId) return;

    const tasksRef = collection(db, `users/${userId}/tasks`);
    addDoc(tasksRef, {
        text: task,
        quadrant: quadrantId,
        createdAt: new Date()
    }).then(() => {
        //console.log("Task added successfully");
    }).catch((error) => {
        console.error("Error adding task: ", error);
    });
}
//Remove task from firestore
function deleteTask(taskId, taskItem) {
    const userId = getCurrentUserId();
    if (!userId) return;

    const taskRef = doc(db, `users/${userId}/tasks/${taskId}`);
    deleteDoc(taskRef).then(() => {
        taskItem.parentNode.removeChild(taskItem);
        console.log("Task deleted successfully");
    }).catch((error) => {
        console.error("Error deleting task: ", error);
    });
}

```

### Step 6: Displaying Tasks in Real Time

Firestore provides real-time listeners, which we’ll use to display tasks as soon as they’re added.

```javascript
auth.onAuthStateChanged(user => {
    if (user) {
        // Listen for changes to the user's task list in Firestore
        db.collection('users').doc(user.uid).collection('tasks')
            .orderBy('timestamp')
            .onSnapshot(snapshot => {
                const taskList = document.getElementById('taskList');
                taskList.innerHTML = '';  // Clear previous tasks
                
                snapshot.forEach(doc => {
                    const taskItem = document.createElement('li');
                    taskItem.textContent = doc.data().task;
                    taskList.appendChild(taskItem);
                });
            });
    }
});

```

### Step 7: Putting It All Together

At this point, we’ve successfully integrated Firebase Authentication and Firestore. Here’s what happens in our app:

1. User Authentication: Users can sign up or log in.
2. Task Management: Once authenticated, users can add tasks to Firestore.
3. Real-Time Updates: Tasks are displayed in real-time, so users can see updates immediately.

## Conclusion

Integrating Firebase Authentication and Firestore into your web application offers a streamlined, serverless solution for managing both user authentication and real-time data storage. In our example, we demonstrated how to build a task management app using the Eisenhower Matrix, but the principles outlined here can easily be extended to a wide range of web applications. Firebase’s suite of tools enables you to focus on your front-end development while it handles the complexities of authentication, database management, and scalability.

What makes Firebase especially appealing is its simplicity and scalability—whether you're working on small projects or building apps with large-scale needs, Firebase can handle it. The free tier is particularly beneficial for smaller applications or prototypes, allowing you to develop and deploy without upfront costs.

If you're looking to expand beyond this, Firebase offers a rich ecosystem of services, including Cloud Functions, Hosting, and Analytics, which can further enhance your app's functionality and performance. These tools, combined with Firebase’s real-time synchronization and seamless cross-platform support, can significantly reduce development time and effort, while ensuring a robust and scalable application.

Explore Firebase’s extensive [documentation](https://firebase.google.com/docs/) to dive deeper into these tools, and take your web app to the next level. Happy coding!
