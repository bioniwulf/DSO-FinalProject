"use strict";

module.exports = {
  types: [
    {value: 'feat',     name: 'A new feature'},
    {value: 'fix',      name: 'A bug fix'},
    {value: 'docs',     name: 'Documentation only changes'},
    {value: 'style',    name: 'Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)'},
    {value: 'refactor', name: 'A code change that neither fixes a bug nor adds a feature'},
    {value: 'perf',     name: 'A code change that improves performance'},
    {value: 'test',     name: 'Adding missing tests'},
    {value: 'chore',    name: 'Changes to the build process or auxiliary tools and libraries such as documentation generation'},
    {value: 'revert',   name: 'Revert to a commit'},
    {value: 'WIP',      name: 'Work in progress'}
  ],

  scopes: [
    { name: 'jupyter' },
    { name: 'python' },
    { name: 'article' },
    { name: 'config' },
    { name: 'doc' }
  ],

  allowTicketNumber: false,
  isTicketNumberRequired: false,
  
  // override the messages, defaults are as follows
  messages: {
    type: "Select the type of change that you're committing:",
    scope: '\nDenote the SCOPE of this change (optional):',
    subject: 'Write a SHORT, IMPERATIVE tense description of the change:\n',
    body: 'Provide a LONGER description of the change (optional). Use "|" to break new line:\n',
    breaking: 'List any BREAKING CHANGES (optional):\n',
    footer: 'List any ISSUES CLOSED by this change (optional). E.g.: #31, #34:\n',
    confirmCommit: 'Are you sure you want to proceed with the commit above?',
  },

  allowBreakingChanges: ['feat', 'fix'],
  // skip any questions you want
  skipQuestions: ['body'],

  // limit subject length
  subjectLimit: 100,
};