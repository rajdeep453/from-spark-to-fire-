"""Message Guard: a small API around the defense middleware.

Adds two things on top of the original project:
  * a persistent store of trusted facts (SQLite) that survives restarts
  * an explain endpoint that shows which trusted fact a claim was compared with
"""
