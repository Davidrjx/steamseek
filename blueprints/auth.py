from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from firebase_config import User, firebase_auth
import uuid
import os
import requests

# Create the blueprint
auth_bp = Blueprint('auth', __name__, template_folder='templates')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle user login via Google OAuth
    """
    error_message = None
    if current_user.is_authenticated:
        return redirect(url_for('search.search_page'))
    
    if request.method == 'POST':
        return redirect(url_for('auth.auth_google'))
        
    return render_template('login.html', error=error_message)

@auth_bp.route('/logout')
@login_required
def logout():
    """
    Handle user logout
    """
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('search.search_page'))

@auth_bp.route('/auth/google')
def auth_google():
    """
    Start the Google OAuth flow by redirecting to Google sign-in page
    """
    # Generate a random state value for CSRF protection
    state = str(uuid.uuid4())
    session['oauth_state'] = state
    
    # Get client ID from environment variables
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    
    # Always use localhost, not 127.0.0.1, to match Google OAuth settings
    #(old style)# redirect_uri = "http://localhost:5000/auth/google/callback"
    redirect_uri = url_for('auth.auth_callback', _external=True)
    
    # Build the Google OAuth URL
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={client_id}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
    )
    
    return redirect(auth_url)

@auth_bp.route('/auth/google/callback')
def auth_callback():
    """
    Handle OAuth callback from Google
    """
    # Verify the state parameter to prevent CSRF
    if 'oauth_state' not in session or request.args.get('state') != session['oauth_state']:
        flash('Invalid authentication state. Please try again.', 'error')
        return redirect(url_for('auth.login'))
    
    # Check for error parameter
    if 'error' in request.args:
        error = request.args.get('error')
        flash(f'Authentication error: {error}', 'error')
        return redirect(url_for('auth.login'))
    
    # Get the authorization code
    code = request.args.get('code')
    if not code:
        flash('No authorization code received', 'error')
        return redirect(url_for('auth.login'))
        
    try:
        # Exchange the code for tokens
        client_id = os.environ.get('GOOGLE_CLIENT_ID')
        client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
        #(old style)# redirect_uri = "http://localhost:5000/auth/google/callback"
        # adaptive redirect_uri based on request host
        redirect_uri = url_for('auth.auth_callback', _external=True)
        
        token_url = "https://oauth2.googleapis.com/token"
        token_payload = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        }
        
        token_response = requests.post(token_url, data=token_payload)
        token_data = token_response.json()
        
        if 'error' in token_data:
            flash(f"Error exchanging code for tokens: {token_data['error']}", 'error')
            return redirect(url_for('auth.login'))
            
        id_token = token_data.get('id_token')
        if not id_token:
            flash('No ID token received', 'error')
            return redirect(url_for('auth.login'))
            
        # Verify the ID token with Firebase
        try:
            firebase_user = firebase_auth.sign_in_with_custom_token(id_token)
            user_info = firebase_user['idToken']
            
            # Extract user information
            uid = user_info.get('localId')
            email = user_info.get('email')
            display_name = user_info.get('displayName')
            photo_url = user_info.get('photoUrl')
            
            # Create or update user in our database
            user = User(uid, email, display_name, photo_url)
            user.create_or_update()
            
            # Log the user in with Flask-Login
            login_user(user)
            flash(f'Welcome, {display_name or email}!', 'success')
            
            # Redirect to the next page or search page
            next_page = session.get('next', url_for('search.search_page'))
            return redirect(next_page)
            
        except Exception as e:
            current_app.logger.error(f"Error verifying ID token: {e}", exc_info=True)
            flash('Error verifying your identity. Please try again.', 'error')
            return redirect(url_for('auth.login'))
            
    except Exception as e:
        current_app.logger.error(f"Error in OAuth callback: {e}", exc_info=True)
        flash('Authentication error. Please try again.', 'error')
        return redirect(url_for('auth.login')) 